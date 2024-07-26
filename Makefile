# Makefile

define HELP_MESSAGE
kscale-nanoppo-library

# Running Tests

1. Run autoformatting: `make format`
2. Run static checks: `make static-checks`

# ------------------------ #
#       Static Checks      #
# ------------------------ #

endef
export HELP_MESSAGE

format:
	@isort --profile black .
	@black .
	@ruff format .
.PHONY: format

static-checks:
	@isort --profile black --check --diff .
	@black --diff --check .
	@ruff check .
	@mypy --install-types --non-interactive .
.PHONY: lint