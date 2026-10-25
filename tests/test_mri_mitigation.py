"""Keep experimental threshold calibration separate from validation/test images."""
import numpy as np
import pandas as pd
from PIL import Image

from src.audit_mri_mitigation import floor_candidate, training_corner_threshold


def test_corner_threshold_reads_unique_training_images_only(tmp_path):
    Image.fromarray(np.full((128,128),10,dtype=np.uint8)).save(tmp_path/'train.png')
    frame = pd.DataFrame([
        dict(path='train.png',split='train',label=0),
        dict(path='train.png',split='train',label=1),
        dict(path='missing_validation.png',split='val',label=0),
        dict(path='missing_test.png',split='test',label=1),
    ])
    threshold,files = training_corner_threshold(tmp_path,[frame])
    assert threshold == 10 and files == 1


def test_floor_candidate_leaves_values_above_training_threshold_unchanged():
    patch = np.array([[0,29/255,30/255,.5,1]],dtype=np.float32)
    result = floor_candidate(patch,29)
    np.testing.assert_array_equal(result,np.array([[0,0,30/255,.5,1]],dtype=np.float32))
    assert patch[0,1] != 0  # Does not mutate the original patch.
