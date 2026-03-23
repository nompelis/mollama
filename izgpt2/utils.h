#ifndef _UTIL_H_
#define _UTIL_H_

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>

void izgpt_util_ConvertEndian( void* out, const void* in, size_t size );

int izgpt_util_DumpSafetensors_header( const char *path );


#endif

