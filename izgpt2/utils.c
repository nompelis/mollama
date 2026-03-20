
#include "utils.h"

//
// Function to convert signed integers (and more) to and from big endian
// to machine endianness
// (We should only be doing up to 8 bytes -- uint64_t)
//

void izgpt_util_ConvertEndian( void* out, const void* in, size_t size )
{
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
   unsigned char buf[8];    // meaningful limit for us...
   unsigned char* p = (unsigned char*) in;
   unsigned char* q = (unsigned char*) out;
   for(int n=0;n<size;++n) buf[n] = p[ size-1 - n ];
   for(int n=0;n<size;++n) q[n] = buf[n];
#else
   for(int n=0;n<size;++n) out[n] = in[n];
#endif
}

