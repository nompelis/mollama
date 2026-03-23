#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include "utils.h"
#include "izgpt_format.h"


void dump_param1( ao_gpt2_t *model )
{
    int H = model->n_embd;
    int V = model->vocab_size;

    // embeddings
    float *v = &model->wte[0 * H];
    FILE *fp = fopen( "EMB_0", "w" );
    for(int i=0;i<H;++i) fprintf( fp, "%.10e\n", v[i] );
    fclose(fp);

    v = &model->wte[1 * H];
    fp = fopen( "EMB_1", "w" );
    for(int i=0;i<H;++i) fprintf( fp, "%.10e\n", v[i] );
    fclose(fp);

    v = &model->wte[(V-1) * H];
    fp = fopen( "EMB_L", "w" );
    for(int i=0;i<H;++i) fprintf( fp, "%.10e\n", v[i] );
    fclose(fp);

    // positional encodings
    v = &model->wpe[0 * H];
    fp = fopen( "POS_0", "w" );
    for(int i=0;i<H;++i) fprintf( fp, "%.10e\n", v[i] );
    fclose(fp);

    v = &model->wpe[1 * H];
    fp = fopen( "POS_1", "w" );
    for(int i=0;i<H;++i) fprintf( fp, "%.10e\n", v[i] );
    fclose(fp);

    v = &model->wpe[(1024-1) * H];
    fp = fopen( "POS_L", "w" );
    for(int i=0;i<H;++i) fprintf( fp, "%.10e\n", v[i] );
    fclose(fp);
}


void dump_param2( ao_gpt2_t *model, int layer_idx )
{
    int H = model->n_embd;
    int F = model->ffn_dim;
    ao_layer_t *l = &( model->layers[layer_idx] );
    ao_attn_t *attn = &( l->attn );;
    ao_ffn_t *ffn = &( l->ffn );;

// Q
    FILE *fp = fopen( "Wq_0", "w" );
    for (int row = 0; row < H; row++) {
        float v = attn->Wq[row * H + 0];
        fprintf( fp, "%.10e\n", v );
    }
    fclose(fp);

    fp = fopen( "Wq_1", "w" );
    for (int row = 0; row < H; row++) {
        float v = attn->Wq[row * H + 1];
        fprintf( fp, "%.10e\n", v );
    }
    fclose(fp);

    fp = fopen( "Wq_L", "w" );
    for (int row = 0; row < H; row++) {
        float v = attn->Wq[row * H + H-1];
        fprintf( fp, "%.10e\n", v );
    }
    fclose(fp);


// K
    fp = fopen( "Wk_0", "w" );
    for (int row = 0; row < H; row++) {
        float v = attn->Wk[row * H + 0];
        fprintf( fp, "%.10e\n", v );
    }
    fclose(fp);

    fp = fopen( "Wk_1", "w" );
    for (int row = 0; row < H; row++) {
        float v = attn->Wk[row * H + 1];
        fprintf( fp, "%.10e\n", v );
    }
    fclose(fp);

    fp = fopen( "Wk_L", "w" );
    for (int row = 0; row < H; row++) {
        float v = attn->Wk[row * H + H-1];
        fprintf( fp, "%.10e\n", v );
    }
    fclose(fp);


// V
    fp = fopen( "Wv_0", "w" );
    for (int row = 0; row < H; row++) {
        float v = attn->Wv[row * H + 0];
        fprintf( fp, "%.10e\n", v );
    }
    fclose(fp);

    fp = fopen( "Wv_1", "w" );
    for (int row = 0; row < H; row++) {
        float v = attn->Wv[row * H + 1];
        fprintf( fp, "%.10e\n", v );
    }
    fclose(fp);

    fp = fopen( "Wv_L", "w" );
    for (int row = 0; row < H; row++) {
        float v = attn->Wv[row * H + H-1];
        fprintf( fp, "%.10e\n", v );
    }
    fclose(fp);


// Wo
    fp = fopen( "Wo_0", "w" );
    for (int row = 0; row < H; row++) {
        float v = attn->Wo[row * H + 0];
        fprintf( fp, "%.10e\n", v );
    }
    fclose(fp);

    fp = fopen( "Wo_1", "w" );
    for (int row = 0; row < H; row++) {
        float v = attn->Wo[row * H + 1];
        fprintf( fp, "%.10e\n", v );
    }
    fclose(fp);

    fp = fopen( "Wo_L", "w" );
    for (int row = 0; row < H; row++) {
        float v = attn->Wo[row * H + H-1];
        fprintf( fp, "%.10e\n", v );
    }
    fclose(fp);


// W1
    fp = fopen( "W1_0", "w" );
    for (int row = 0; row < F; row++) {
        float v = ffn->W1[row * H + 0];
        fprintf( fp, "%.10e\n", v );
    }
    fclose(fp);

    fp = fopen( "W1_1", "w" );
    for (int row = 0; row < F; row++) {
        float v = ffn->W1[row * H + 1];
        fprintf( fp, "%.10e\n", v );
    }
    fclose(fp);

    fp = fopen( "W1_L", "w" );
    for (int row = 0; row < F; row++) {
        float v = ffn->W1[row * H + H-1];
        fprintf( fp, "%.10e\n", v );
    }
    fclose(fp);


// W2
    fp = fopen( "W2_0", "w" );
    for (int row = 0; row < H; row++) {
        float v = ffn->W2[row * F + 0];
        fprintf( fp, "%.10e\n", v );
    }
    fclose(fp);

    fp = fopen( "W2_1", "w" );
    for (int row = 0; row < H; row++) {
        float v = ffn->W2[row * F + 1];
        fprintf( fp, "%.10e\n", v );
    }
    fclose(fp);

    fp = fopen( "W2_L", "w" );
    for (int row = 0; row < H; row++) {
        float v = ffn->W2[row * F + F-1];
        fprintf( fp, "%.10e\n", v );
    }
    fclose(fp);
}


void dump_param3( ao_gpt2_t *model, int layer_idx )
{
    int H = model->n_embd;
    int F = model->ffn_dim;
    ao_layer_t *l = &( model->layers[layer_idx] );
    ao_attn_t *attn = &( l->attn );;
    ao_ffn_t *ffn = &( l->ffn );;

// Q
    FILE *fp = fopen( "bq", "w" );
    for (int row = 0; row < H; row++) {
        fprintf( fp, "%.10e\n", attn->bq[row] );
    }
    fclose(fp);

// K
    fp = fopen( "bk", "w" );
    for (int row = 0; row < H; row++) {
        fprintf( fp, "%.10e\n", attn->bk[row] );
    }
    fclose(fp);

// V
    fp = fopen( "bv", "w" );
    for (int row = 0; row < H; row++) {
        fprintf( fp, "%.10e\n", attn->bv[row] );
    }
    fclose(fp);

// Wo
    fp = fopen( "bo", "w" );
    for (int row = 0; row < H; row++) {
        fprintf( fp, "%.10e\n", attn->bo[row] );
    }
    fclose(fp);

// FF_1
    fp = fopen( "b1", "w" );
    for (int row = 0; row < F; row++) {
        fprintf( fp, "%.10e\n", ffn->b1[row] );
    }
    fclose(fp);

// FF_2
    fp = fopen( "b2", "w" );
    for (int row = 0; row < H; row++) {
        fprintf( fp, "%.10e\n", ffn->b2[row] );
    }
    fclose(fp);

// LN_1
    fp = fopen( "LN1g", "w" );
    for (int row = 0; row < H; row++) {
        fprintf( fp, "%.10e\n", l->ln1.gamma[row] );
    }
    fclose(fp);
    fp = fopen( "LN1b", "w" );
    for (int row = 0; row < H; row++) {
        fprintf( fp, "%.10e\n", l->ln1.beta[row] );
    }
    fclose(fp);

// LN_2
    fp = fopen( "LN2g", "w" );
    for (int row = 0; row < H; row++) {
        fprintf( fp, "%.10e\n", l->ln2.gamma[row] );
    }
    fclose(fp);
    fp = fopen( "LN2b", "w" );
    for (int row = 0; row < H; row++) {
        fprintf( fp, "%.10e\n", l->ln2.beta[row] );
    }
    fclose(fp);

// LN_Final
    fp = fopen( "LNfg", "w" );
    for (int row = 0; row < H; row++) {
        fprintf( fp, "%.10e\n", model->ln_f.gamma[row] );
    }
    fclose(fp);
    fp = fopen( "LNfb", "w" );
    for (int row = 0; row < H; row++) {
        fprintf( fp, "%.10e\n", model->ln_f.beta[row] );
    }
    fclose(fp);
}


#ifdef _DRIVER_
int main( int argc, char *argv[] )
{
    if (argc < 3) {
        printf("Supply arguments: '%s <num> <string>'\n", argv[0]);
        return 0;
    }

    if (!strcmp(argv[1], "0")) {    // display the "safetensors" JSON header
        return izgpt_util_DumpSafetensors_header( argv[2] );
    }

    if (!strcmp(argv[1], "1") || !strcmp(argv[1], "2")) {   // load a model
        ao_gpt2_t *model = load_model(argv[2]);

        if (model)
        if (!strcmp(argv[2], "2")) {    // dump data on the screen
            // dump_param1( model );
            // dump_param2( model, 11 );
            // dump_param3( model, 0 );
        }

        if (model) destroy_model(model);
    }

    if (!strcmp(argv[1], "3")) {    // become a shmem process
        sm_gpt2_t *sm = smload_model(argv[2]);
        if (sm) fprintf( stdout, " [Info]  Shmem segment with ID: %d\n",sm->id);

        while(1) sleep(1000);

        if (sm) { free(sm->model); free(sm); }    // clean-up (never happens)
    }

    if (!strcmp(argv[1], "4")) {    // attach to a shmem
        int id = atoi(argv[2]);
        sm_gpt2_t *sm = smattach_model(id);
        if (sm) fprintf( stdout, " [Info]  Shmem segment with ID: %d\n",sm->id);
    }

    return 0;
}
#endif

