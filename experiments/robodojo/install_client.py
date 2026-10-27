"""Install only a new simulator policy registry; existing policies are untouched."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stack', required=True)
    args = parser.parse_args()
    stack = Path(args.stack).resolve()
    if not (stack / 'XPolicyLab/policy/StarWAMFastWAM/deploy.py').is_file():
        raise ValueError('Expected the existing audited RoboDojo eval-stack')
    source = Path(__file__).resolve().parent / 'xpolicy_client'
    target = stack / 'XPolicyLab/policy/FastWAMAnchorHelios'
    names = ('__init__.py', 'deploy.py', 'deploy.yml')
    for name in names:
        path = target / name
        if path.exists() and path.read_bytes() != (source / name).read_bytes():
            raise ValueError(f'Existing registry differs; refusing to overwrite {path}')
    target.mkdir(exist_ok=True)
    for name in names:
        (target / name).write_bytes((source / name).read_bytes())
    print(f'Installed simulator registry: {target}')


if __name__ == '__main__':
    main()
