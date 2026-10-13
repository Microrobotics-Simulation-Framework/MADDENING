#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(void){
  const char *v[] = {"NaN","Infinity","-Infinity","nan","NAN","inf","INF","Inf",
                     "infinity","-inf","+Infinity","nan(0x1)","NAN(quiet)","0x1p3",
                     "  1.5"," Infinity","1e5","NaN ","1.0e","","true","null"};
  for (size_t i=0;i<sizeof v/sizeof*v;i++){
    char *end; double d = strtod(v[i], &end);
    printf("%-12s -> consumed=%2d value=%g %s\n", v[i], (int)(end-v[i]), d,
           end==v[i] ? "(NOTHING PARSED)" : (*end ? "(TRAILING JUNK)" : ""));
  }
  return 0;
}
