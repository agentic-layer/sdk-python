.PHONY: all
all: build


.PHONY: build
build:
	$(MAKE) -C shared build
	$(MAKE) -C adk build
	$(MAKE) -C msaf build


.PHONY: test
test: build
	$(MAKE) -C shared test
	$(MAKE) -C adk test
	$(MAKE) -C msaf test


.PHONY: check
check: build
	$(MAKE) -C shared check
	$(MAKE) -C adk check
	$(MAKE) -C msaf check


.PHONY: check-fix
check-fix: build
	$(MAKE) -C shared check-fix
	$(MAKE) -C adk check-fix
	$(MAKE) -C msaf check-fix


.PHONY: update
update:
	$(MAKE) -C shared update
	$(MAKE) -C adk update
	$(MAKE) -C msaf update
