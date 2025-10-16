# Makefile
CC ?= cc
CFLAGS ?= -O3 -fPIC -Wall -Wextra -Werror -std=c11
LDFLAGS_SHARED ?= -shared
AR ?= ar
RANLIB ?= ranlib

SRC = negamax_bb.c
OBJ = $(SRC:.c=.o)

TESTER      := negamax_bb_tester
TESTER_SRC  := negamax_bb_tester.c
TESTER_OBJ  := $(TESTER_SRC:.c=.o)

PROBE      := negamax_bb_probe
PROBE_SRC  := negamax_bb_probe.c
PROBE_OBJ  := $(PROBE_SRC:.c=.o)

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  SO  := libnegamaxbb.dylib
  LDFLAGS_SHARED += -Wl,-install_name,@rpath/$(SO)
else
  SO  := libnegamaxbb.so
endif
STATIC := libnegamaxbb.a

.PHONY: all clean
all: $(SO) $(TESTER) $(PROBE)

$(SO): $(OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS_SHARED) -o $@ $^

$(STATIC): $(OBJ)
	$(AR) rcs $@ $^
	$(RANLIB) $@

negamax_bb.o: negamax_bb.c negamax_bb.h
	$(CC) $(CFLAGS) -DNEGAMAXBB_BUILD -c $< -o $@

$(TESTER_OBJ): $(TESTER_SRC) negamax_bb.h
	$(CC) $(CFLAGS) -c $< -o $@

$(TESTER): $(TESTER_OBJ) $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^

$(PROBE_OBJ): $(PROBE_SRC) negamax_bb.h
	$(CC) $(CFLAGS) -c $< -o $@

$(PROBE): $(PROBE_OBJ) $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm -f $(OBJ) $(SO) $(STATIC) $(TESTER_OBJ) $(TESTER) $(PROBE_OBJ) $(PROBE)
