
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


//
// Function to read and display the JSON header of a "safetensors" file
//

int izgpt_util_DumpSafetensors_header( const char *path )
{
   FILE *f = fopen(path, "rb");
   if (!f) {
      fprintf( stdout, " [Error]  Could not open file: \"%s\"\n", path );
      perror("fopen");
      return 1;
   }

   // --- Step 1: read header length (uint64_t little-endian) ---
   uint64_t header_len=0;
   if (fread(&header_len, sizeof(uint64_t), 1, f) != 1) {
      fprintf( stdout, " [Error]  Failed to read header length\n" );
      fclose(f);
      return 2;
   }

   // sanity check
   if (header_len == 0 || header_len > (1ULL << 30)) {
      fprintf( stdout, " [Error]  Suspicious header length: %lu\n",
               (unsigned long) header_len );
      fclose(f);
      return 3;
   }

   // --- Step 2: peek first byte of JSON ---
   int c = fgetc(f);
   if (c != '{') {
      fprintf( stdout, " [Error]  Invalid safetensors file: JSON does not start with '{'\n" );
      fclose(f);
      return 4;
   }

   // rewind one byte so we read full JSON
   fseek(f, -1, SEEK_CUR);

   // --- Step 3: read JSON header ---
   char *json = (char *) malloc(header_len + 1);
   if (!json) {
      fprintf( stdout, " [Error]  Could not malloc() %ld bytes for header\n",
               header_len+1 );
      fclose(f);
      return -1;
   }

   if (fread(json, 1, header_len, f) != header_len) {
      fprintf( stdout, " [Error]  Failed to read JSON header\n" );
      free(json);
      fclose(f);
      return 5;
   }

   json[header_len] = '\0';

   // --- Step 4: simple indentation printer ---
   int indent = 0;
   int in_string = 0;

   for (uint64_t i = 0; i < header_len; i++) {
      char ch = json[i];

      if (ch == '"' && (i == 0 || json[i - 1] != '\\')) {
          in_string = !in_string;
          putchar(ch);
          continue;
      }

      if (!in_string) {
          if (ch == '{' || ch == '[') {
             putchar(ch);
             putchar('\n');
             indent++;
             for (int j = 0; j < indent; j++) printf("  ");
             continue;
          }
          else if (ch == '}' || ch == ']') {
             putchar('\n');
             indent--;
             for (int j = 0; j < indent; j++) printf("  ");
             putchar(ch);
             continue;
          }
          else if (ch == ',') {
             putchar(ch);
             putchar('\n');
             for (int j = 0; j < indent; j++) printf("  ");
             continue;
          }
          else if (ch == ':') {
             printf(": ");
             continue;
          }
      }

      putchar(ch);
   }

   putchar('\n');

   free(json);
   fclose(f);
   return 0;
}

