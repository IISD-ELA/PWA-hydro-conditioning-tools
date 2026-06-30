# Test targets for pwa-tools.
#
#   make test         unit tests (mirrors GitHub CI)
#   make integration  module-CLI e2e on real data (skips without PWA_STEP0_CONFIG)
#   make regression   WhiteboxTools + gdal end-to-end vs the grassmere baseline
#                      (skips without the reference dataset)
#
# See tests/integration/README.md and tests/regression/ for inputs.
.PHONY: test integration regression

test:
	pytest tests/unit -m "not integration and not regression and not slow"

integration:
	pytest tests/integration

regression:
	pytest tests/regression -m regression
