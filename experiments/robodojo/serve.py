"""Serve one Anchor-Helios policy using the existing XPolicyLab transport."""
import argparse
import asyncio
import json
import logging
import os

from client_server.ws.model_server import PolicyServer, PolicyServerConfig
from experiments.robodojo.xpolicy_model import Model
from experiments.robodojo.idle_server import IdlePolicyServer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment', required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8096)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    with open(args.deployment) as handle:
        model = Model(json.load(handle))
    asyncio.run(IdlePolicyServer(model, PolicyServerConfig(host=args.host, port=args.port),
        first_timeout=float(os.getenv('POLICY_FIRST_REQUEST_TIMEOUT_SECONDS','7200')),
        idle_timeout=float(os.getenv('POLICY_IDLE_TIMEOUT_SECONDS','900'))).serve_forever())


if __name__ == '__main__':
    main()
