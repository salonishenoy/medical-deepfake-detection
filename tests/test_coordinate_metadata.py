"""Check coordinate metadata against actual resized pixels, not just its formula."""
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from src.config import DataConfig
from src.data_loader import BTDDataset
from src.mri_data_loader import MRIDataConfig, MRIDataset


def make_dataset(tmp_path, modality, raw, x, y, output_size, orientation='preserve'):
    path = tmp_path/('image.npy' if modality == 'CT' else 'image.png')
    if modality == 'CT':
        np.save(path,raw)
    else:
        Image.fromarray(raw).save(path)
    frame = pd.DataFrame([dict(path=path.name,scan_id='scan',scanner='scanner',cur_slice=1,
                               task='injection',label=1,x=x,y=y)])
    if modality == 'CT':
        return BTDDataset(frame,tmp_path,DataConfig(image_size=output_size,imagenet_norm=False))
    return MRIDataset(frame,tmp_path,MRIDataConfig(image_size=output_size,imagenet_norm=False,
                                                  orientation_policy=orientation))


@pytest.mark.parametrize('modality', ['CT','MRI'])
def test_default_resize_metadata_points_to_original_pixel_value(tmp_path,modality):
    # A local linear intensity plane lets us check the point against real output pixels.
    yy,xx = np.indices((256,256))
    raw = (-700+xx+yy).astype(np.float32) if modality == 'CT' else ((xx+yy)%256).astype(np.uint8)
    dataset = make_dataset(tmp_path,modality,raw,90,100,224)
    image,_,meta = dataset[0]
    grid = torch.tensor([[[[2*(meta['x']+.5)/224-1,2*(meta['y']+.5)/224-1]]]])
    actual = F.grid_sample(image[None],grid,mode='bilinear',align_corners=False)[0,0,0,0]
    expected = (raw[100,90]+700)/2700 if modality == 'CT' else raw[100,90]/255
    assert float(actual) == pytest.approx(float(expected),abs=1e-6)


@pytest.mark.parametrize('modality,native_size', [('CT',96),('MRI',128)])
def test_marked_pixel_centroid_matches_metadata_after_twofold_resize(tmp_path,modality,native_size):
    raw = np.full((256,256),-700,dtype=np.float32) if modality == 'CT' else np.zeros((256,256),dtype=np.uint8)
    raw[100,90] = 2000 if modality == 'CT' else 255
    dataset = make_dataset(tmp_path,modality,raw,90,100,2*native_size)
    image,_,meta = dataset[0]
    weights = image[0]
    axis = torch.arange(2*native_size,dtype=torch.float32)
    center_x = float((weights.sum(dim=0)*axis).sum()/weights.sum())
    center_y = float((weights.sum(dim=1)*axis).sum()/weights.sum())
    assert meta['x'] == pytest.approx(center_x,abs=1e-6)
    assert meta['y'] == pytest.approx(center_y,abs=1e-6)


@pytest.mark.parametrize('modality,native_size', [('CT',96),('MRI',128)])
@pytest.mark.parametrize('x,y', [(0,0),(255,255),(0,255),(255,0)])
def test_downsized_edge_coordinates_round_trip_without_clamping(tmp_path,modality,native_size,x,y):
    raw = np.zeros((256,256),dtype=np.float32 if modality == 'CT' else np.uint8)
    image,_,meta = make_dataset(tmp_path,modality,raw,x,y,native_size//2)[0]
    scale = .5
    recovered_x = (meta['x']+.5)/scale-.5+meta['crop_left']
    recovered_y = (meta['y']+.5)/scale-.5+meta['crop_top']
    assert recovered_x == x and recovered_y == y
    # The full image area includes half a pixel outside its first/last centers.
    assert -.5 <= meta['x'] <= image.shape[-1]-.5
    assert -.5 <= meta['y'] <= image.shape[-2]-.5
    assert meta['x'] == (-.25 if x == 0 else native_size/2-.75)
    assert meta['y'] == (-.25 if y == 0 else native_size/2-.75)


def test_mri_reflection_then_resize_tracks_source_pixel(tmp_path):
    yy,xx = np.indices((256,256))
    raw = ((xx+2*yy)%256).astype(np.uint8)
    raw[:30],raw[-30:] = 255,0  # Force the explicit reference policy to reflect.
    image,_,meta = make_dataset(tmp_path,'MRI',raw,90,100,224,'author_border')[0]
    assert meta['orientation_flipped'] and meta['oriented_y'] == 155
    grid = torch.tensor([[[[2*(meta['x']+.5)/224-1,2*(meta['y']+.5)/224-1]]]])
    actual = F.grid_sample(image[None],grid,mode='bilinear',align_corners=False)[0,0,0,0]
    assert float(actual) == pytest.approx(float(raw[100,90]/255),abs=1e-6)
