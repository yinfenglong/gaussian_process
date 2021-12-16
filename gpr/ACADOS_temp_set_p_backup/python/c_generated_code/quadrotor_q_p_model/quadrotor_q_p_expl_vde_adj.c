/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) quadrotor_q_p_expl_vde_adj_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_sq CASADI_PREFIX(sq)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

casadi_real casadi_sq(casadi_real x) { return x*x;}

static const casadi_int casadi_s0[14] = {10, 1, 0, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
static const casadi_int casadi_s1[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s2[7] = {3, 1, 0, 3, 0, 1, 2};
static const casadi_int casadi_s3[18] = {14, 1, 0, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};

/* quadrotor_q_p_expl_vde_adj:(i0[10],i1[10],i2[4],i3[3])->(o0[14]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a3, a4, a5, a6, a7, a8, a9;
  a0=0.;
  if (res[0]!=0) res[0][0]=a0;
  if (res[0]!=0) res[0][1]=a0;
  if (res[0]!=0) res[0][2]=a0;
  a0=arg[0]? arg[0][3] : 0;
  a1=(a0+a0);
  a2=arg[2]? arg[2][3] : 0;
  a3=arg[1]? arg[1][9] : 0;
  a4=(a2*a3);
  a1=(a1*a4);
  a5=arg[0]? arg[0][4] : 0;
  a6=2.;
  a7=arg[1]? arg[1][8] : 0;
  a8=(a2*a7);
  a8=(a6*a8);
  a9=(a5*a8);
  a1=(a1-a9);
  a9=arg[0]? arg[0][5] : 0;
  a10=arg[1]? arg[1][7] : 0;
  a2=(a2*a10);
  a2=(a6*a2);
  a11=(a9*a2);
  a1=(a1+a11);
  a11=arg[2]? arg[2][2] : 0;
  a12=5.0000000000000000e-01;
  a13=arg[1]? arg[1][6] : 0;
  a13=(a12*a13);
  a14=(a11*a13);
  a1=(a1+a14);
  a14=arg[2]? arg[2][1] : 0;
  a15=arg[1]? arg[1][5] : 0;
  a15=(a12*a15);
  a16=(a14*a15);
  a1=(a1+a16);
  a16=arg[2]? arg[2][0] : 0;
  a17=arg[1]? arg[1][4] : 0;
  a17=(a12*a17);
  a18=(a16*a17);
  a1=(a1+a18);
  if (res[0]!=0) res[0][3]=a1;
  a1=arg[0]? arg[0][6] : 0;
  a18=(a1*a2);
  a19=(a5+a5);
  a19=(a19*a4);
  a20=(a0*a8);
  a19=(a19+a20);
  a18=(a18-a19);
  a19=(a14*a13);
  a18=(a18+a19);
  a19=(a11*a15);
  a18=(a18-a19);
  a19=arg[1]? arg[1][3] : 0;
  a12=(a12*a19);
  a19=(a16*a12);
  a18=(a18-a19);
  if (res[0]!=0) res[0][4]=a18;
  a18=(a1*a8);
  a19=(a9+a9);
  a19=(a19*a4);
  a18=(a18-a19);
  a19=(a0*a2);
  a18=(a18+a19);
  a19=(a16*a13);
  a18=(a18-a19);
  a19=(a11*a17);
  a18=(a18+a19);
  a19=(a14*a12);
  a18=(a18-a19);
  if (res[0]!=0) res[0][5]=a18;
  a18=(a1+a1);
  a18=(a18*a4);
  a8=(a9*a8);
  a18=(a18+a8);
  a2=(a5*a2);
  a18=(a18+a2);
  a16=(a16*a15);
  a18=(a18+a16);
  a14=(a14*a17);
  a18=(a18-a14);
  a11=(a11*a12);
  a18=(a18-a11);
  if (res[0]!=0) res[0][6]=a18;
  a18=arg[1]? arg[1][0] : 0;
  if (res[0]!=0) res[0][7]=a18;
  a18=arg[1]? arg[1][1] : 0;
  if (res[0]!=0) res[0][8]=a18;
  a18=arg[1]? arg[1][2] : 0;
  if (res[0]!=0) res[0][9]=a18;
  a18=(a1*a15);
  a11=(a9*a13);
  a18=(a18-a11);
  a11=(a0*a17);
  a18=(a18+a11);
  a11=(a5*a12);
  a18=(a18-a11);
  if (res[0]!=0) res[0][10]=a18;
  a18=(a5*a13);
  a11=(a0*a15);
  a18=(a18+a11);
  a11=(a1*a17);
  a18=(a18-a11);
  a11=(a9*a12);
  a18=(a18-a11);
  if (res[0]!=0) res[0][11]=a18;
  a13=(a0*a13);
  a15=(a5*a15);
  a13=(a13-a15);
  a17=(a9*a17);
  a13=(a13+a17);
  a12=(a1*a12);
  a13=(a13-a12);
  if (res[0]!=0) res[0][12]=a13;
  a13=casadi_sq(a0);
  a12=casadi_sq(a5);
  a13=(a13-a12);
  a12=casadi_sq(a9);
  a13=(a13-a12);
  a12=casadi_sq(a1);
  a13=(a13+a12);
  a13=(a13*a3);
  a3=(a9*a1);
  a12=(a0*a5);
  a3=(a3-a12);
  a3=(a6*a3);
  a3=(a3*a7);
  a13=(a13+a3);
  a0=(a0*a9);
  a5=(a5*a1);
  a0=(a0+a5);
  a6=(a6*a0);
  a6=(a6*a10);
  a13=(a13+a6);
  if (res[0]!=0) res[0][13]=a13;
  return 0;
}

CASADI_SYMBOL_EXPORT int quadrotor_q_p_expl_vde_adj(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int quadrotor_q_p_expl_vde_adj_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int quadrotor_q_p_expl_vde_adj_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void quadrotor_q_p_expl_vde_adj_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int quadrotor_q_p_expl_vde_adj_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void quadrotor_q_p_expl_vde_adj_release(int mem) {
}

CASADI_SYMBOL_EXPORT void quadrotor_q_p_expl_vde_adj_incref(void) {
}

CASADI_SYMBOL_EXPORT void quadrotor_q_p_expl_vde_adj_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int quadrotor_q_p_expl_vde_adj_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int quadrotor_q_p_expl_vde_adj_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real quadrotor_q_p_expl_vde_adj_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* quadrotor_q_p_expl_vde_adj_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* quadrotor_q_p_expl_vde_adj_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* quadrotor_q_p_expl_vde_adj_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    case 3: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* quadrotor_q_p_expl_vde_adj_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int quadrotor_q_p_expl_vde_adj_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
