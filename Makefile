# Test configuration
PYTEST = pytest
PYTEST_OPTS = -v --tb=short
COVERAGE_OPTS = --cov=drdmannturb --cov-report=term-missing --cov-report=html

# Test directories
UNIT_TESTS = test/unit_tests
INTEGRATION_TESTS = test/integration_tests
ALL_TESTS = test

# Default target
.PHONY: test
test: test-unit

# =============================================================================
# UNIT TESTS
# =============================================================================

.PHONY: test-unit
test-unit:
	$(PYTEST) $(PYTEST_OPTS) $(UNIT_TESTS)

.PHONY: test-unit-nn
test-unit-nn:
	$(PYTEST) $(PYTEST_OPTS) $(UNIT_TESTS)/nn_modules/

.PHONY: test-unit-spectra
test-unit-spectra:
	$(PYTEST) $(PYTEST_OPTS) $(UNIT_TESTS)/spectra_fitting/

.PHONY: test-unit-fluctuation
test-unit-fluctuation:
	$(PYTEST) $(PYTEST_OPTS) $(UNIT_TESTS)/fluctuation_generation/

.PHONY: test-unit-parameters
test-unit-parameters:
	$(PYTEST) $(PYTEST_OPTS) $(UNIT_TESTS)/parameters/

# =============================================================================
# INTEGRATION TESTS
# =============================================================================

.PHONY: test-integration
test-integration:
	$(PYTEST) $(PYTEST_OPTS) $(INTEGRATION_TESTS)

.PHONY: test-integration-cross
test-integration-cross:
	$(PYTEST) $(PYTEST_OPTS) $(INTEGRATION_TESTS)/cross_module/

.PHONY: test-integration-spectra
test-integration-spectra:
	$(PYTEST) $(PYTEST_OPTS) $(INTEGRATION_TESTS)/spectra_fitting/

# =============================================================================
# SPECIFIC MODULE TESTS
# =============================================================================

.PHONY: test-taunet
test-taunet:
	$(PYTEST) $(PYTEST_OPTS) -k "taunet"

.PHONY: test-rational
test-rational:
	$(PYTEST) $(PYTEST_OPTS) -k "rational"

.PHONY: test-spectral-tensor
test-spectral-tensor:
	$(PYTEST) $(PYTEST_OPTS) -k "spectral_tensor"

.PHONY: test-eddy-lifetime
test-eddy-lifetime:
	$(PYTEST) $(PYTEST_OPTS) -k "eddy_lifetime"

.PHONY: test-energy-spectrum
test-energy-spectrum:
	$(PYTEST) $(PYTEST_OPTS) -k "energy_spectrum"

.PHONY: test-precision
test-precision:
	$(PYTEST) $(PYTEST_OPTS) --group=precision

.PHONY: test-precision-float32
test-precision-float32:
	$(PYTEST) $(PYTEST_OPTS) --group=precision --precision=float32

.PHONY: test-precision-float64
test-precision-float64:
	$(PYTEST) $(PYTEST_OPTS) --group=precision --precision=float64

# =============================================================================
# COVERAGE TESTS
# =============================================================================

.PHONY: test-cov
test-cov:
	$(PYTEST) $(PYTEST_OPTS) $(COVERAGE_OPTS) $(ALL_TESTS)

.PHONY: test-cov-unit
test-cov-unit:
	$(PYTEST) $(PYTEST_OPTS) $(COVERAGE_OPTS) $(UNIT_TESTS)

.PHONY: test-cov-integration
test-cov-integration:
	$(PYTEST) $(PYTEST_OPTS) $(COVERAGE_OPTS) $(INTEGRATION_TESTS)

# =============================================================================
# SLOW TESTS
# =============================================================================

.PHONY: test-slow
test-slow:
	$(PYTEST) $(PYTEST_OPTS) --runslow $(ALL_TESTS)

.PHONY: test-slow-unit
test-slow-unit:
	$(PYTEST) $(PYTEST_OPTS) --runslow $(UNIT_TESTS)

# =============================================================================
# DEBUGGING AND DEVELOPMENT
# =============================================================================

.PHONY: test-debug
test-debug:
	$(PYTEST) $(PYTEST_OPTS) -s --tb=long $(ALL_TESTS)

.PHONY: test-failed
test-failed:
	$(PYTEST) $(PYTEST_OPTS) --lf

.PHONY: test-last-failed
test-last-failed:
	$(PYTEST) $(PYTEST_OPTS) --lf

.PHONY: test-collect
test-collect:
	$(PYTEST) --collect-only $(ALL_TESTS)

.PHONY: test-collect-unit
test-collect-unit:
	$(PYTEST) --collect-only $(UNIT_TESTS)

# =============================================================================
# PLATFORM SPECIFIC TESTS
# =============================================================================

.PHONY: test-cpu
test-cpu:
	$(PYTEST) $(PYTEST_OPTS) --platform=cpu $(ALL_TESTS)

.PHONY: test-gpu
test-gpu:
	$(PYTEST) $(PYTEST_OPTS) --platform=gpu $(ALL_TESTS)

# =============================================================================
# COMPREHENSIVE TEST SUITES
# =============================================================================

.PHONY: test-all
test-all: test-unit test-integration

.PHONY: test-all-with-slow
test-all-with-slow: test-unit test-integration test-slow

.PHONY: test-all-with-cov
test-all-with-cov: test-cov-unit test-cov-integration

.PHONY: test-core
test-core: test-unit-nn test-unit-spectra test-unit-parameters

# =============================================================================
# CLEANUP
# =============================================================================

.PHONY: clean
clean:
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

.PHONY: clean-tests
clean-tests:
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage

# =============================================================================
# HELP
# =============================================================================

.PHONY: help
help:
	@echo "Available test targets:"
	@echo ""
	@echo "Unit Tests:"
	@echo "  test-unit              - Run all unit tests"
	@echo "  test-unit-nn           - Run nn_modules tests"
	@echo "  test-unit-spectra      - Run spectra_fitting tests"
	@echo "  test-unit-fluctuation  - Run fluctuation_generation tests"
	@echo "  test-unit-parameters   - Run parameters tests"
	@echo ""
	@echo "Integration Tests:"
	@echo "  test-integration       - Run all integration tests"
	@echo "  test-integration-cross - Run cross_module tests"
	@echo "  test-integration-spectra - Run spectra integration tests"
	@echo ""
	@echo "Specific Module Tests:"
	@echo "  test-taunet            - Run TauNet tests"
	@echo "  test-rational          - Run Rational tests"
	@echo "  test-spectral-tensor   - Run spectral tensor tests"
	@echo "  test-eddy-lifetime     - Run eddy lifetime tests"
	@echo "  test-energy-spectrum   - Run energy spectrum tests"
	@echo ""
	@echo "Precision Tests:"
	@echo "  test-precision         - Run all precision tests"
	@echo "  test-precision-float32 - Run float32 precision tests"
	@echo "  test-precision-float64 - Run float64 precision tests"
	@echo ""
	@echo "Coverage Tests:"
	@echo "  test-cov               - Run all tests with coverage"
	@echo "  test-cov-unit          - Run unit tests with coverage"
	@echo "  test-cov-integration   - Run integration tests with coverage"
	@echo ""
	@echo "Slow Tests:"
	@echo "  test-slow              - Run all slow tests"
	@echo "  test-slow-unit         - Run slow unit tests"
	@echo ""
	@echo "Debugging:"
	@echo "  test-debug             - Run tests with debug output"
	@echo "  test-failed            - Run only failed tests"
	@echo "  test-collect           - Show what tests would run"
	@echo ""
	@echo "Platform Tests:"
	@echo "  test-cpu               - Run CPU-only tests"
	@echo "  test-gpu               - Run GPU tests"
	@echo ""
	@echo "Comprehensive Suites:"
	@echo "  test-all               - Run unit + integration tests"
	@echo "  test-all-with-slow     - Run all tests including slow ones"
	@echo "  test-all-with-cov      - Run all tests with coverage"
	@echo "  test-core              - Run core module tests"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean                  - Clean all build artifacts"
	@echo "  clean-tests            - Clean test artifacts only"
	@echo ""
	@echo "Help:"
	@echo "  help                   - Show this help message"
