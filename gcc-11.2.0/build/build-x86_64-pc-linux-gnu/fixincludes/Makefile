# Makefile for fixincludes.
#
#   Copyright (C) 1998, 1999, 2000, 2001, 2003, 2004, 2009, 2010, 2012
#   Free Software Foundation, Inc.

#This file is part of fixincludes.

#fixincludes is free software; you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation; either version 3, or (at your option)
#any later version.

#fixincludes is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with fixincludes; see the file COPYING3.  If not see
#<http://www.gnu.org/licenses/>.

SHELL=/bin/bash

# Some versions of `touch' (such as the version on Solaris 2.8) 
# do not correctly set the timestamp due to buggy versions of `utime'
# in the kernel.  So, we use `echo' instead. 
STAMP = echo timestamp >

CC = gcc
CFLAGS = -g -O2
WARN_CFLAGS = -W -Wall -Wwrite-strings -Wstrict-prototypes -Wmissing-prototypes -Wold-style-definition -Wmissing-format-attribute -Wno-overlength-strings -pedantic -Wno-long-long 
LDFLAGS = 
INCLUDES = -I. -I$(srcdir) -I../include -I$(srcdir)/../include
FIXINC_CFLAGS = -DHAVE_CONFIG_H $(INCLUDES)

# Directory where sources are, from where we are.
srcdir = ../../../fixincludes
VPATH = $(srcdir)

# Directory in which to put the directories used by the compiler.
libdir = ${exec_prefix}/lib
# Directory in which GCC puts its executables.
libexecdir = ${exec_prefix}/libexec

# End of variables for you to override.

# The target that we're configured for.
target = x86_64-pc-linux-gnu
target_noncanonical:=x86_64-pc-linux-gnu

# The version of GCC in this tree
gcc_version := $(shell cat $(srcdir)/../gcc/BASE-VER)

# Directory in which the compiler finds libraries etc.
libsubdir = $(libdir)/gcc/$(target_noncanonical)/$(gcc_version)
# Directory in which the compiler finds executables
libexecsubdir = $(libexecdir)/gcc/$(target_noncanonical)/$(gcc_version)
# Where our executable files go
itoolsdir = $(libexecsubdir)/install-tools
# Where our data files go
itoolsdatadir = $(libsubdir)/install-tools

# Locate mkinstalldirs.
mkinstalldirs=$(SHELL) $(srcdir)/../mkinstalldirs

AUTOCONF = autoconf
AUTOHEADER = autoheader
ACLOCAL = aclocal
ACLOCAL_AMFLAGS = -I .. -I ../config

default : all

# Now figure out from those variables how to compile and link.

.c.o:
	$(CC) -c $(CFLAGS) $(WARN_CFLAGS) $(CPPFLAGS) $(FIXINC_CFLAGS) $<

# The only suffixes we want for implicit rules are .c and .o.
.SUFFIXES:
.SUFFIXES: .c .o

#

## # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
##
##  Makefile for constructing the "best" include fixer we can
##
## # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

LIBIBERTY=../libiberty/libiberty.a

ALLOBJ = fixincl.o fixtests.o fixfixes.o server.o procopen.o \
      fixlib.o fixopts.o

TESTOBJ = fixincl.o fixlib.o fixtests.o fixopts.o
FIXOBJ  = fixfixes.o fixlib.o fixopts.o

HDR = server.h fixlib.h
FI  = fixincl
AF  = applyfix

all : oneprocess fixinc.sh mkheaders
gen : $(srcdir)/fixincl.x


oneprocess : full-stamp
twoprocess : test-stamp $(AF)

full-stamp : $(ALLOBJ) $(LIBIBERTY)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(FI) $(ALLOBJ) $(LIBIBERTY)
	$(STAMP) $@

test-stamp : $(TESTOBJ) $(LIBIBERTY)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(FI) $(TESTOBJ) $(LIBIBERTY)
	$(STAMP) $@

$(AF): $(FIXOBJ) $(LIBIBERTY)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(FIXOBJ) $(LIBIBERTY)

$(ALLOBJ)   : $(HDR)
fixincl.o   : fixincl.c  $(srcdir)/fixincl.x
fixtests.o  : fixtests.c
fixfixes.o  : fixfixes.c $(srcdir)/fixincl.x
server.o    : server.c
procopen.o  : procopen.c
fixlib.o    : fixlib.c

fixinc.sh : fixinc.in mkfixinc.sh Makefile
	srcdir="$(srcdir)" $(SHELL) $(srcdir)/mkfixinc.sh $(target)

$(srcdir)/fixincl.x: # fixincl.tpl inclhack.def
	cd $(srcdir) ; $(SHELL) ./genfixes

mostlyclean : 
	rm -f *.o *-stamp $(AF) $(FI) *~ fixinc.sh

clean: mostlyclean
	rm -f mkheaders mkheaders.almost

distclean: clean
	rm -f Makefile config.h config.log config.status stamp-h

maintainer-clean: distclean
	rm -f $(srcdir)/fixincl.x

distclean : clean

Makefile: $(srcdir)/Makefile.in config.status
	$(SHELL) ./config.status Makefile

mkheaders.almost: $(srcdir)/mkheaders.in config.status
	CONFIG_FILES=mkheaders.almost:mkheaders.in \
	CONFIG_HEADERS= ./config.status

mkheaders: mkheaders.almost $(srcdir)/../gcc/BASE-VER
	sed -e 's/@gcc_version@/$(gcc_version)/' < $< > $@T
	mv -f $@T $@

config.h: stamp-h
stamp-h: $(srcdir)/config.h.in config.status
	$(SHELL) ./config.status config.h

config.status: $(srcdir)/configure
	$(SHELL) ./config.status --recheck

$(srcdir)/configure: # $(srcdir)/configure.ac $(srcdir)/aclocal.m4
	cd $(srcdir) && $(AUTOCONF)

$(srcdir)/config.h.in: # $(srcdir)/configure.ac
	cd $(srcdir) && $(AUTOHEADER)

$(srcdir)/aclocal.m4: # $(srcdir)/configure.ac
	cd $(srcdir) && $(ACLOCAL) $(ACLOCAL_AMFLAGS)

check : all
	autogen -T $(srcdir)/check.tpl $(srcdir)/inclhack.def
	$(SHELL) ./check.sh $(srcdir)/tests/base
	@rm -f ./check.sh

install : all
	-rm -rf $(DESTDIR)$(itoolsdir)
	$(mkinstalldirs) $(DESTDIR)$(itoolsdir)
	$(mkinstalldirs) $(DESTDIR)$(itoolsdatadir)/include
	$(INSTALL_DATA) $(srcdir)/README-fixinc \
	  $(DESTDIR)$(itoolsdatadir)/include/README
	$(INSTALL_SCRIPT) fixinc.sh $(DESTDIR)$(itoolsdir)/fixinc.sh
	$(INSTALL_PROGRAM) $(FI) $(DESTDIR)$(itoolsdir)/$(FI)
	$(INSTALL_SCRIPT) mkheaders $(DESTDIR)$(itoolsdir)/mkheaders

install-strip: install
	test -z '$(STRIP)' || $(STRIP) $(DESTDIR)$(itoolsdir)/$(FI)

.PHONY: all check install install-strip
.PHONY: dvi pdf info html install-pdf install-info install-html

dvi :
pdf :
info :
html :
install-pdf :
install-info :
install-html :
installcheck :
