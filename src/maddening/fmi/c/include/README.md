# Vendored FMI 3.0 headers

`fmi3Functions.h`, `fmi3FunctionTypes.h` and `fmi3PlatformTypes.h` are the
official FMI 3.0 C headers published by the Modelica Association (copyright
2008-2011 MODELISAR consortium, 2012-2022 Modelica Association Project "FMI"),
licensed under the 2-Clause BSD License as stated in each file.  They are
vendored unmodified so `maddening.fmi.package.build_fmu_binary` can compile
`../maddening_fmu.c` with nothing but a C compiler.
