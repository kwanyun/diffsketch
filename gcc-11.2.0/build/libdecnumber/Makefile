# Makefile.  Generated from Makefile.in by configure.
# Makefile for libdecnumber.  Run 'configure' to generate Makefile from Makefile.in

# Copyright (C) 2005-2021 Free Software Foundation, Inc.

#This file is part of GCC.

#GCC is free software; you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation; either version 3, or (at your option)
#any later version.

#GCC is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with GCC; see the file COPYING3.  If not see
#<http://www.gnu.org/licenses/>.



srcdir = ../../libdecnumber
top_builddir = .
VPATH = ../../libdecnumber
INSTALL = @INSTALL@
AR = ar
ARFLAGS = cru
ACLOCAL = aclocal
AUTOCONF = autoconf
AUTOHEADER = autoheader
CC = gcc
CFLAGS = -g
WARN_CFLAGS = -W -Wall -Wwrite-strings -Wstrict-prototypes -Wmissing-prototypes -Wold-style-definition -Wmissing-format-attribute -Wcast-qual -pedantic -Wno-long-long 
CPP = gcc -E
CPPFLAGS = 
INSTALL_DATA = @INSTALL_DATA@
INSTALL_PROGRAM = @INSTALL_PROGRAM@
LDFLAGS = -static-libstdc++ -static-libgcc 
LIBICONV = @LIBICONV@
PACKAGE = @PACKAGE@
PICFLAG = 
CET_HOST_FLAGS = 
RANLIB = ranlib
SHELL = /bin/bash
objext = o
NOLTO_FLAGS = -fno-lto

datarootdir = ${prefix}/share
datadir = ${datarootdir}
exec_prefix = /data/kwan/gcc-11.2.0/local
libdir = ${exec_prefix}/lib
localedir = $(datadir)/locale
prefix = /data/kwan/gcc-11.2.0/local

ADDITIONAL_OBJS =  $(bid_OBJS)

enable_decimal_float= bid

INCLUDES = -I$(srcdir) -I.

ALL_CFLAGS = $(CFLAGS) $(WARN_CFLAGS) $(NOLTO_FLAGS) $(INCLUDES) $(CPPFLAGS) \
	     $(PICFLAG) $(CET_HOST_FLAGS)

bid_OBJS = bid2dpd_dpd2bid.$(objext) host-ieee32.$(objext) \
	host-ieee64.$(objext) host-ieee128.$(objext)

libdecnumber_a_OBJS = decNumber.$(objext) decContext.$(objext) \
	decimal32.$(objext) decimal64.$(objext) decimal128.$(objext) \
	$(ADDITIONAL_OBJS)

libdecnumber_a_SOURCES = decContext.c decContext.h decDPD.h \
	decNumber.c decNumber.h decNumberLocal.h \
	decContextSymbols.h decDPDSymbols.h decNumberSymbols.h \
	dpd/decimal128.c dpd/decimal128.h dpd/decimal128Symbols.h \
	dpd/decimal32.c dpd/decimal32.h dpd/decimal32Symbols.h \
	dpd/decimal64.c dpd/decimal64.h dpd/decimal64Symbols.h \
	bid/decimal128.c bid/decimal128.h \
	bid/decimal32.c bid/decimal32.h \
	bid/decimal64.c bid/decimal64.h

all: libdecnumber.a

.SUFFIXES:
.SUFFIXES: .c .o .obj

libdecnumber.a: $(libdecnumber_a_OBJS)
	-rm -f $@
	$(AR) $(ARFLAGS) $@ $(libdecnumber_a_OBJS)
	$(RANLIB) $@

# Rules to rebuild the configuration

Makefile: $(srcdir)/Makefile.in config.status
	$(SHELL) ./config.status Makefile

config.status: $(srcdir)/configure
	$(SHELL) ./config.status --recheck

$(srcdir)/configure: # $(srcdir)/aclocal.m4
	cd $(srcdir) && $(AUTOCONF)

aclocal_deps = \
	$(srcdir)/../config/stdint.m4 \
	$(srcdir)/../config/warnings.m4 \
	$(srcdir)/../config/override.m4 \
	$(srcdir)/../config/dfp.m4 \
	$(srcdir)/configure.ac

$(srcdir)/aclocal.m4: # $(aclocal_deps)
	cd $(srcdir) && $(ACLOCAL) -I ../config

config.h: stamp-h1
	test -f config.h || (rm -f stamp-h1 && $(MAKE) stamp-h1)

stamp-h1: $(srcdir)/config.in config.status
	-rm -f stamp-h1
	$(SHELL) ./config.status config.h

$(srcdir)/config.in: # $(srcdir)/configure
	cd $(srcdir) && $(AUTOHEADER)
	-rm -f stamp-h1

# Dependencies.

decContext.$(objext): decContext.c decContext.h decNumberLocal.h \
	decContextSymbols.h
decNumber.$(objext):  decNumber.c decNumber.h decContext.h decNumberLocal.h \
	decNumberSymbols.h
decimal32.$(objext):  $(srcdir)/$(enable_decimal_float)/decimal32.c \
   $(srcdir)/$(enable_decimal_float)/decimal32.h \
   $(srcdir)/$(enable_decimal_float)/decimal32Symbols.h \
   decNumber.h decContext.h decNumberLocal.h
	$(COMPILE) $(srcdir)/$(enable_decimal_float)/decimal32.c
decimal64.$(objext):  $(srcdir)/$(enable_decimal_float)/decimal64.c \
   $(srcdir)/$(enable_decimal_float)/decimal64.h \
   $(srcdir)/$(enable_decimal_float)/decimal64Symbols.h \
   decNumber.h decContext.h decNumberLocal.h
	$(COMPILE) $(srcdir)/$(enable_decimal_float)/decimal64.c
decimal128.$(objext):  $(srcdir)/$(enable_decimal_float)/decimal128.c \
   $(srcdir)/$(enable_decimal_float)/decimal128.h \
   $(srcdir)/$(enable_decimal_float)/decimal128Symbols.h\
   $(srcdir)/$(enable_decimal_float)/decimal128Local.h\
   decNumber.h decContext.h decNumberLocal.h 
	$(COMPILE) $(srcdir)/$(enable_decimal_float)/decimal128.c
bid2dpd_dpd2bid.$(objext) : $(srcdir)/bid/bid2dpd_dpd2bid.c $(srcdir)/bid/bid2dpd_dpd2bid.h
	$(COMPILE) $(srcdir)/bid/bid2dpd_dpd2bid.c
host-ieee32.$(objext) : $(srcdir)/bid/host-ieee32.c $(srcdir)/bid/decimal32.h
	$(COMPILE) $(srcdir)/bid/host-ieee32.c
host-ieee64.$(objext) : $(srcdir)/bid/host-ieee64.c $(srcdir)/bid/decimal64.h
	$(COMPILE) $(srcdir)/bid/host-ieee64.c
host-ieee128.$(objext) : $(srcdir)/bid/host-ieee128.c $(srcdir)/bid/decimal128.h
	$(COMPILE) $(srcdir)/bid/host-ieee128.c
# Other miscellaneous targets.

mostlyclean:
	-rm -f *.$(objext)

clean: mostlyclean
	-rm -rf libdecnumber.a $(srcdir)/autom4te.cache

distclean: clean
	-rm -f config.h stamp-h1 config.status config.cache config.log \
	  configure.lineno configure.status.lineno Makefile localedir.h \
	  localedir.hs gstdint.h

maintainer-clean: distclean
	@echo "This command is intended for maintainers to use"
	@echo "it deletes files that may require special tools to rebuild."
	-rm -f $(srcdir)/configure $(srcdir)/aclocal.m4

check:
installcheck:
dvi:
pdf:
html:
info:
install-info:
install-pdf:
install-man:
install-html:
install:

.PHONY: installdirs install install-strip mostlyclean clean distclean \
  maintainer-clean check installcheck dvi pdf html info install-info \
  install-pdf install-man update-po install-html

COMPILE = source='$<' object='$@' libtool=no $(CC) $(DEFS) $(INCLUDES) $(CPPFLAGS) $(ALL_CFLAGS) -c

# Implicit rules

.c.$(objext):
	$(COMPILE) $<

# Tell versions [3.59,3.63) of GNU make to not export all variables.
# Otherwise a system limit (for SysV at least) may be exceeded.
.NOEXPORT:
