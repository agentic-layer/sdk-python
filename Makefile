.PHONY: all
all: build


.PHONY: build
build:
	$(MAKE) -C adk build
	$(MAKE) -C msaf build


.PHONY: test
test: build
	$(MAKE) -C adk test
	$(MAKE) -C msaf test


.PHONY: check
check: build
	$(MAKE) -C adk check
	$(MAKE) -C msaf check


.PHONY: check-fix
check-fix: build
	$(MAKE) -C adk check-fix
	$(MAKE) -C msaf check-fix
