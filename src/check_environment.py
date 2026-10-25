"""Check the platform-scoped reproduction environment using standard-library metadata.

This does not install packages, change RNG state, or modify preprocessing.
"""
import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import sys

REPO = Path(__file__).resolve().parents[1]


def canonical(name):
    return re.sub(r'[-_.]+','-',name).lower()


def check(lock, contract_path):
    expected = {}
    for line in lock.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        match = re.fullmatch(r'([A-Za-z0-9_.-]+)==([^\s;]+)',line)
        if not match:
            raise ValueError(f'Expected an exact name==version pin: {line}')
        name,version = match.groups()
        if canonical(name) in expected:
            raise ValueError(f'Duplicate lock entry: {name}')
        expected[canonical(name)] = version
    if not expected:
        raise ValueError('Empty environment lock')
    actual = {canonical(d.metadata['Name']):d.version for d in metadata.distributions()}
    issues = []
    required_python = (REPO/'.python-version').read_text().strip()
    if platform.python_version() != required_python:
        issues.append(f'Python must match audited version {required_python}')
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        issues.append('This lock was verified for macOS arm64 only; this platform needs separate validation')
    venv_cfg = Path(sys.prefix)/'pyvenv.cfg'
    isolated = (sys.prefix != sys.base_prefix and venv_cfg.exists()
                and bool(re.search(r'include-system-site-packages\s*=\s*false',venv_cfg.read_text(),re.I)))
    if not isolated:
        issues.append('Use an isolated virtual environment without system site packages')
    for name,version in expected.items():
        if actual.get(name) != version:
            issues.append(f'{name}: expected {version}, found {actual.get(name,"missing")}')
        elif not Path(metadata.distribution(name).locate_file('')).resolve().is_relative_to(Path(sys.prefix).resolve()):
            issues.append(f'{name}: distribution is outside this environment')
    contract = json.loads(contract_path.read_text())
    for name,digest in contract['provenance']['source_sha256'].items():
        if hashlib.sha256((REPO/name).read_bytes()).hexdigest() != digest:
            issues.append(f'Frozen preprocessing source changed: {name}')
    return dict(passed=not issues,python=platform.python_version(),implementation=platform.python_implementation(),
        platform=platform.platform(),system=platform.system(),machine=platform.machine(),
        isolated_environment=isolated,locked_packages=len(expected),expected_packages=expected,
        matched_package_count=sum(actual.get(k)==v for k,v in expected.items()),
        extra_packages={k:v for k,v in actual.items() if k not in expected},issues=issues,
        lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest(),
        contract_id=contract['contract_id'],contract_sha256=hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        frozen_preprocessing_source_sha256=contract['provenance']['source_sha256'],
        scope='Same-platform package/version reproduction check; no cross-platform or full-training guarantee')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock',type=Path,default=REPO/'requirements-macos-arm64-py312.lock')
    parser.add_argument('--contract',type=Path,default=REPO/'docs/preprocessing_contract_v1.json')
    parser.add_argument('--out',type=Path,default=Path('results/environment.json'))
    args=parser.parse_args()
    try:
        report=check(args.lock,args.contract)
    except (OSError,ValueError,KeyError) as error:
        report=dict(passed=False,issues=[str(error)])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['passed','python','locked_packages','matched_package_count','issues'] if k in report},indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
