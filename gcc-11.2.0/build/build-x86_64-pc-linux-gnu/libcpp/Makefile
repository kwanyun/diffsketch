# Makefile.  Generated from Makefile.in by configure.
# Makefile for libcpp.  Run 'configure' to generate Makefile from Makefile.in

# Copyright (C) 2004-2021 Free Software Foundation, Inc.

#This file is part of libcpp.

#libcpp is free software; you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation; either version 3, or (at your option)
#any later version.

#libcpp is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with libcpp; see the file COPYING3.  If not see
#<http://www.gnu.org/licenses/>.



srcdir = ../../../libcpp
top_builddir = .
VPATH = ../../../libcpp
INSTALL = /bin/install -c
AR = ar
ARFLAGS = cru
ACLOCAL = aclocal
AUTOCONF = autoconf
AUTOHEADER = autoheader
CATALOGS = $(patsubst %,po/%,)
CC = gcc
CFLAGS = -g -O2
WARN_CFLAGS = -W -Wall -Wno-narrowing -Wwrite-strings -Wmissing-format-attribute -Wstrict-prototypes -Wmissing-prototypes -Wold-style-definition -Wc++-compat -pedantic -Wno-long-long 
CXX = g++ -std=c++11
CXXFLAGS = -g -O2
WARN_CXXFLAGS = -W -Wall -Wno-narrowing -Wwrite-strings -Wmissing-format-attribute -pedantic -Wno-long-long 
CPP = gcc -E
CPPFLAGS = 
EXEEXT = 
GMSGFMT = 
INCINTL = 
INSTALL_DATA = /bin/install -c -m 644
INSTALL_PROGRAM = /bin/install -c
INSTALL_SCRIPT = /bin/install -c
LDFLAGS = 
LIBICONV = 
LIBINTL = 
PACKAGE = cpplib
RANLIB = ranlib
SHELL = /bin/bash
USED_CATALOGS = 
XGETTEXT = 
CCDEPMODE = @CCDEPMODE@
CXXDEPMODE = depmode=gcc3
DEPDIR = .deps
NOEXCEPTION_FLAGS = -fno-exceptions -fno-rtti
PICFLAG = 
CET_HOST_FLAGS = 

datarootdir = ${prefix}/share
datadir = ${datarootdir}
exec_prefix = /data/kwan/gcc-11.2.0/local
libdir = ${exec_prefix}/lib
localedir = $(datadir)/locale
prefix = /data/kwan/gcc-11.2.0/local

MSGMERGE = msgmerge
mkinstalldirs = $(SHELL) $(srcdir)/../mkinstalldirs
depcomp = $(SHELL) $(srcdir)/../depcomp

INCLUDES = -I$(srcdir) -I. -I$(srcdir)/../include  \
	-I$(srcdir)/include

ALL_CFLAGS = $(CFLAGS) $(WARN_CFLAGS) $(INCLUDES) $(CPPFLAGS) $(PICFLAG) \
	$(CET_HOST_FLAGS)
ALL_CXXFLAGS = $(CXXFLAGS) $(WARN_CXXFLAGS) $(NOEXCEPTION_FLAGS) $(INCLUDES) \
	$(CPPFLAGS) $(PICFLAG) $(CET_HOST_FLAGS)

# The name of the compiler to use.
COMPILER = $(CXX)
COMPILER_FLAGS = $(ALL_CXXFLAGS)
DEPMODE = $(CXXDEPMODE)


libcpp_a_OBJS = charset.o directives.o errors.o \
	expr.o files.o identifiers.o init.o lex.o line-map.o macro.o \
	mkdeps.o pch.o symtab.o traditional.o

libcpp_a_SOURCES = charset.c directives.c errors.c \
	expr.c files.c identifiers.c init.c lex.c line-map.c macro.c \
	mkdeps.c pch.c symtab.c traditional.c

all: libcpp.a $(USED_CATALOGS)

.SUFFIXES:
.SUFFIXES: .c .gmo .o .obj .po .pox

libcpp.a: $(libcpp_a_OBJS)
	-rm -f libcpp.a
	$(AR) $(ARFLAGS) libcpp.a $(libcpp_a_OBJS)
	$(RANLIB) libcpp.a

# Rules to rebuild the configuration

Makefile: $(srcdir)/Makefile.in config.status
	$(SHELL) ./config.status Makefile

config.status: $(srcdir)/configure
	$(SHELL) ./config.status --recheck

$(srcdir)/configure: # $(srcdir)/aclocal.m4
	cd $(srcdir) && $(AUTOCONF)

$(srcdir)/aclocal.m4: # $(srcdir)/../config/acx.m4 \
	$(srcdir)/../config/gettext-sister.m4 $(srcdir)/../config/iconv.m4 \
	$(srcdir)/../config/codeset.m4 $(srcdir)/../config/lib-ld.m4 \
	$(srcdir)/../config/lib-link.m4 $(srcdir)/../config/lib-prefix.m4 \
	$(srcdir)/../config/override.m4 $(srcdir)/../config/proginstall.m4 \
	$(srcdir)/configure.ac
	cd $(srcdir) && $(ACLOCAL) -I ../config

config.h: stamp-h1
	test -f config.h || (rm -f stamp-h1 && $(MAKE) stamp-h1)

stamp-h1: $(srcdir)/config.in config.status
	-rm -f stamp-h1
	$(SHELL) ./config.status config.h

$(srcdir)/config.in: # $(srcdir)/configure.ac
	cd $(srcdir) && $(AUTOHEADER)
	-rm -f stamp-h1

# It is not possible to get LOCALEDIR defined in config.h because
# the value it needs to be defined to is only determined in the
# Makefile.  Hence we do this instead.
localedir.h: localedir.hs; @true
localedir.hs: Makefile
	echo "#define LOCALEDIR \"$(localedir)\"" > localedir.new
	$(srcdir)/../move-if-change localedir.new localedir.h
	echo timestamp > localedir.hs

# Installation rules and other phony targets

# These rule has to look for .gmo modules in both srcdir and
# the cwd, and has to check that we actually have a catalog
# for each language, in case they weren't built or included
# with the distribution.
installdirs:
	@$(mkinstalldirs) $(DESTDIR)$(datadir); \
	cats="$(CATALOGS)"; for cat in $$cats; do \
	  lang=`basename $$cat | sed 's/\.gmo$$//'`; \
	  if [ -f $$cat ] || [ -f $(srcdir)/$$cat ]; then \
	    dir=$(localedir)/$$lang/LC_MESSAGES; \
	    $(mkinstalldirs) $(DESTDIR)$$dir || exit 1; \
	  fi; \
	done

install-strip install: all installdirs
	cats="$(CATALOGS)"; for cat in $$cats; do \
	  lang=`basename $$cat | sed 's/\.gmo$$//'`; \
	  if [ -f $$cat ]; then :; \
	  elif [ -f $(srcdir)/$$cat ]; then cat=$(srcdir)/$$cat; \
	  else continue; \
	  fi; \
	  dir=$(localedir)/$$lang/LC_MESSAGES; \
	  echo $(INSTALL_DATA) $$cat $(DESTDIR)$$dir/$(PACKAGE).mo; \
	  $(INSTALL_DATA) $$cat $(DESTDIR)$$dir/$(PACKAGE).mo; \
	done

mostlyclean:
	-rm -f *.o

clean: mostlyclean
	-rm -rf libcpp.a $(srcdir)/autom4te.cache

distclean: clean
	-rm -f config.h stamp-h1 config.status config.cache config.log \
	  configure.lineno configure.status.lineno Makefile localedir.h \
	  localedir.hs $(DEPDIR)/*.Po
	-rmdir $(DEPDIR)

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

update-po: $(CATALOGS:.gmo=.pox)

.PHONY: installdirs install install-strip mostlyclean clean distclean \
  maintainer-clean check installcheck dvi pdf html info install-info \
  install-man update-po install-html

# Dependency rule.
COMPILE.base = $(COMPILER) $(DEFS) $(INCLUDES) $(CPPFLAGS) $(COMPILER_FLAGS) -c
ifeq ($(DEPMODE),depmode=gcc3)
# Note that we put the dependencies into a .Tpo file, then move them
# into place if the compile succeeds.  We need this because gcc does
# not atomically write the dependency output file.
COMPILE = $(COMPILE.base) -o $@ -MT $@ -MMD -MP -MF $(DEPDIR)/$*.Tpo
POSTCOMPILE = @mv $(DEPDIR)/$*.Tpo $(DEPDIR)/$*.Po
else
COMPILE = source='$<' object='$@' libtool=no DEPDIR=$(DEPDIR) $(DEPMODE) \
	  $(depcomp) $(COMPILE.base)
# depcomp handles atomicity for us, so we don't need a postcompile
# step.
POSTCOMPILE =
endif

# Implicit rules and I18N

.c.o:
	$(COMPILE) $<
	$(POSTCOMPILE)

# N.B. We do not attempt to copy these into $(srcdir).
.po.gmo:
	$(mkinstalldirs) po
	$(GMSGFMT) --statistics -o $@ $<

# The new .po has to be gone over by hand, so we deposit it into
# build/po with a different extension.
# If build/po/$(PACKAGE).pot exists, use it (it was just created),
# else use the one in srcdir.
.po.pox:
	$(mkinstalldirs) po
	$(MSGMERGE) $< `if test -f po/$(PACKAGE).pot; \
	                then echo po/$(PACKAGE).pot; \
	                else echo $(srcdir)/po/$(PACKAGE).pot; fi` -o $@

# Rule for regenerating the message template.
$(PACKAGE).pot: po/$(PACKAGE).pot
po/$(PACKAGE).pot: $(libcpp_a_SOURCES)
	$(mkinstalldirs) $(srcdir)/po
	$(XGETTEXT) --default-domain=$(PACKAGE) \
	  --keyword=_ --keyword=N_ \
	  --keyword=cpp_error:3 \
	  --keyword=cpp_warning:3 \
	  --keyword=cpp_pedwarning:3 \
	  --keyword=cpp_warning_syshdr:3 \
	  --keyword=cpp_error_with_line:5 \
	  --keyword=cpp_warning_with_line:5 \
	  --keyword=cpp_pedwarning_with_line:5 \
	  --keyword=cpp_warning_with_line_syshdr:5 \
	  --keyword=cpp_errno:3 \
	  --keyword=SYNTAX_ERROR --keyword=SYNTAX_ERROR2 \
	  --copyright-holder="Free Software Foundation, Inc." \
	  --msgid-bugs-address="https://gcc.gnu.org/bugs/" \
	  --language=c -o po/$(PACKAGE).pot.tmp $^
	sed 's:$(srcdir)/::g' <po/$(PACKAGE).pot.tmp >po/$(PACKAGE).pot
	rm po/$(PACKAGE).pot.tmp

TAGS_SOURCES = $(libcpp_a_SOURCES) internal.h system.h ucnid.h \
    include/cpplib.h include/line-map.h include/mkdeps.h include/symtab.h

TAGS: $(TAGS_SOURCES)
	cd $(srcdir) && etags $(TAGS_SOURCES)

# Tell versions [3.59,3.63) of GNU make to not export all variables.
# Otherwise a system limit (for SysV at least) may be exceeded.
.NOEXPORT:

# Dependencies
-include $(patsubst %.o, $(DEPDIR)/%.Po, $(libcpp_a_OBJS))

# Dependencies on generated headers have to be explicit.
init.o: localedir.h
