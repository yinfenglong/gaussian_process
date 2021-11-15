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
  #define CASADI_PREFIX(ID) quadrotor_q_p_expl_vde_forw_ ## ID
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
#define casadi_s4 CASADI_PREFIX(s4)
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
static const casadi_int casadi_s1[113] = {10, 10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
static const casadi_int casadi_s2[47] = {10, 4, 0, 10, 20, 30, 40, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
static const casadi_int casadi_s3[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s4[7] = {3, 1, 0, 3, 0, 1, 2};

/* quadrotor_q_p_expl_vde_forw:(i0[10],i1[10x10],i2[10x4],i3[4],i4[3])->(o0[10],o1[10x10],o2[10x4]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a3, a4, a5, a6, a7, a8, a9;
  a0=arg[0]? arg[0][7] : 0;
  if (res[0]!=0) res[0][0]=a0;
  a0=arg[0]? arg[0][8] : 0;
  if (res[0]!=0) res[0][1]=a0;
  a0=arg[0]? arg[0][9] : 0;
  if (res[0]!=0) res[0][2]=a0;
  a0=5.0000000000000000e-01;
  a1=arg[3]? arg[3][0] : 0;
  a2=arg[0]? arg[0][4] : 0;
  a3=(a1*a2);
  a4=arg[3]? arg[3][1] : 0;
  a5=arg[0]? arg[0][5] : 0;
  a6=(a4*a5);
  a3=(a3+a6);
  a6=arg[3]? arg[3][2] : 0;
  a7=arg[0]? arg[0][6] : 0;
  a8=(a6*a7);
  a3=(a3+a8);
  a3=(a0*a3);
  a3=(-a3);
  if (res[0]!=0) res[0][3]=a3;
  a3=arg[0]? arg[0][3] : 0;
  a8=(a1*a3);
  a9=(a6*a5);
  a8=(a8+a9);
  a9=(a4*a7);
  a8=(a8-a9);
  a8=(a0*a8);
  if (res[0]!=0) res[0][4]=a8;
  a8=(a4*a3);
  a9=(a6*a2);
  a8=(a8-a9);
  a9=(a1*a7);
  a8=(a8+a9);
  a8=(a0*a8);
  if (res[0]!=0) res[0][5]=a8;
  a8=(a6*a3);
  a9=(a4*a2);
  a8=(a8+a9);
  a9=(a1*a5);
  a8=(a8-a9);
  a8=(a0*a8);
  if (res[0]!=0) res[0][6]=a8;
  a8=2.;
  a9=(a3*a5);
  a10=(a2*a7);
  a9=(a9+a10);
  a9=(a8*a9);
  a10=arg[3]? arg[3][3] : 0;
  a11=(a9*a10);
  a12=arg[4]? arg[4][0] : 0;
  a11=(a11+a12);
  if (res[0]!=0) res[0][7]=a11;
  a11=(a5*a7);
  a12=(a3*a2);
  a11=(a11-a12);
  a11=(a8*a11);
  a12=(a11*a10);
  a13=arg[4]? arg[4][1] : 0;
  a12=(a12+a13);
  if (res[0]!=0) res[0][8]=a12;
  a12=casadi_sq(a3);
  a13=casadi_sq(a2);
  a12=(a12-a13);
  a13=casadi_sq(a5);
  a12=(a12-a13);
  a13=casadi_sq(a7);
  a12=(a12+a13);
  a13=(a12*a10);
  a14=9.8065999999999995e+00;
  a13=(a13-a14);
  a14=arg[4]? arg[4][2] : 0;
  a13=(a13+a14);
  a14=2.8999999999999998e-01;
  a13=(a13-a14);
  if (res[0]!=0) res[0][9]=a13;
  a13=arg[1]? arg[1][7] : 0;
  if (res[1]!=0) res[1][0]=a13;
  a13=arg[1]? arg[1][8] : 0;
  if (res[1]!=0) res[1][1]=a13;
  a13=arg[1]? arg[1][9] : 0;
  if (res[1]!=0) res[1][2]=a13;
  a13=arg[1]? arg[1][4] : 0;
  a14=(a1*a13);
  a15=arg[1]? arg[1][5] : 0;
  a16=(a4*a15);
  a14=(a14+a16);
  a16=arg[1]? arg[1][6] : 0;
  a17=(a6*a16);
  a14=(a14+a17);
  a14=(a0*a14);
  a14=(-a14);
  if (res[1]!=0) res[1][3]=a14;
  a14=arg[1]? arg[1][3] : 0;
  a17=(a1*a14);
  a18=(a6*a15);
  a17=(a17+a18);
  a18=(a4*a16);
  a17=(a17-a18);
  a17=(a0*a17);
  if (res[1]!=0) res[1][4]=a17;
  a17=(a4*a14);
  a18=(a6*a13);
  a17=(a17-a18);
  a18=(a1*a16);
  a17=(a17+a18);
  a17=(a0*a17);
  if (res[1]!=0) res[1][5]=a17;
  a17=(a6*a14);
  a18=(a4*a13);
  a17=(a17+a18);
  a18=(a1*a15);
  a17=(a17-a18);
  a17=(a0*a17);
  if (res[1]!=0) res[1][6]=a17;
  a17=(a5*a14);
  a18=(a3*a15);
  a17=(a17+a18);
  a18=(a7*a13);
  a19=(a2*a16);
  a18=(a18+a19);
  a17=(a17+a18);
  a17=(a8*a17);
  a17=(a10*a17);
  if (res[1]!=0) res[1][7]=a17;
  a17=(a7*a15);
  a18=(a5*a16);
  a17=(a17+a18);
  a18=(a2*a14);
  a19=(a3*a13);
  a18=(a18+a19);
  a17=(a17-a18);
  a17=(a8*a17);
  a17=(a10*a17);
  if (res[1]!=0) res[1][8]=a17;
  a17=(a3+a3);
  a14=(a17*a14);
  a18=(a2+a2);
  a13=(a18*a13);
  a14=(a14-a13);
  a13=(a5+a5);
  a15=(a13*a15);
  a14=(a14-a15);
  a15=(a7+a7);
  a16=(a15*a16);
  a14=(a14+a16);
  a14=(a10*a14);
  if (res[1]!=0) res[1][9]=a14;
  a14=arg[1]? arg[1][17] : 0;
  if (res[1]!=0) res[1][10]=a14;
  a14=arg[1]? arg[1][18] : 0;
  if (res[1]!=0) res[1][11]=a14;
  a14=arg[1]? arg[1][19] : 0;
  if (res[1]!=0) res[1][12]=a14;
  a14=arg[1]? arg[1][14] : 0;
  a16=(a1*a14);
  a19=arg[1]? arg[1][15] : 0;
  a20=(a4*a19);
  a16=(a16+a20);
  a20=arg[1]? arg[1][16] : 0;
  a21=(a6*a20);
  a16=(a16+a21);
  a16=(a0*a16);
  a16=(-a16);
  if (res[1]!=0) res[1][13]=a16;
  a16=arg[1]? arg[1][13] : 0;
  a21=(a1*a16);
  a22=(a6*a19);
  a21=(a21+a22);
  a22=(a4*a20);
  a21=(a21-a22);
  a21=(a0*a21);
  if (res[1]!=0) res[1][14]=a21;
  a21=(a4*a16);
  a22=(a6*a14);
  a21=(a21-a22);
  a22=(a1*a20);
  a21=(a21+a22);
  a21=(a0*a21);
  if (res[1]!=0) res[1][15]=a21;
  a21=(a6*a16);
  a22=(a4*a14);
  a21=(a21+a22);
  a22=(a1*a19);
  a21=(a21-a22);
  a21=(a0*a21);
  if (res[1]!=0) res[1][16]=a21;
  a21=(a5*a16);
  a22=(a3*a19);
  a21=(a21+a22);
  a22=(a7*a14);
  a23=(a2*a20);
  a22=(a22+a23);
  a21=(a21+a22);
  a21=(a8*a21);
  a21=(a10*a21);
  if (res[1]!=0) res[1][17]=a21;
  a21=(a7*a19);
  a22=(a5*a20);
  a21=(a21+a22);
  a22=(a2*a16);
  a23=(a3*a14);
  a22=(a22+a23);
  a21=(a21-a22);
  a21=(a8*a21);
  a21=(a10*a21);
  if (res[1]!=0) res[1][18]=a21;
  a16=(a17*a16);
  a14=(a18*a14);
  a16=(a16-a14);
  a19=(a13*a19);
  a16=(a16-a19);
  a20=(a15*a20);
  a16=(a16+a20);
  a16=(a10*a16);
  if (res[1]!=0) res[1][19]=a16;
  a16=arg[1]? arg[1][27] : 0;
  if (res[1]!=0) res[1][20]=a16;
  a16=arg[1]? arg[1][28] : 0;
  if (res[1]!=0) res[1][21]=a16;
  a16=arg[1]? arg[1][29] : 0;
  if (res[1]!=0) res[1][22]=a16;
  a16=arg[1]? arg[1][24] : 0;
  a20=(a1*a16);
  a19=arg[1]? arg[1][25] : 0;
  a14=(a4*a19);
  a20=(a20+a14);
  a14=arg[1]? arg[1][26] : 0;
  a21=(a6*a14);
  a20=(a20+a21);
  a20=(a0*a20);
  a20=(-a20);
  if (res[1]!=0) res[1][23]=a20;
  a20=arg[1]? arg[1][23] : 0;
  a21=(a1*a20);
  a22=(a6*a19);
  a21=(a21+a22);
  a22=(a4*a14);
  a21=(a21-a22);
  a21=(a0*a21);
  if (res[1]!=0) res[1][24]=a21;
  a21=(a4*a20);
  a22=(a6*a16);
  a21=(a21-a22);
  a22=(a1*a14);
  a21=(a21+a22);
  a21=(a0*a21);
  if (res[1]!=0) res[1][25]=a21;
  a21=(a6*a20);
  a22=(a4*a16);
  a21=(a21+a22);
  a22=(a1*a19);
  a21=(a21-a22);
  a21=(a0*a21);
  if (res[1]!=0) res[1][26]=a21;
  a21=(a5*a20);
  a22=(a3*a19);
  a21=(a21+a22);
  a22=(a7*a16);
  a23=(a2*a14);
  a22=(a22+a23);
  a21=(a21+a22);
  a21=(a8*a21);
  a21=(a10*a21);
  if (res[1]!=0) res[1][27]=a21;
  a21=(a7*a19);
  a22=(a5*a14);
  a21=(a21+a22);
  a22=(a2*a20);
  a23=(a3*a16);
  a22=(a22+a23);
  a21=(a21-a22);
  a21=(a8*a21);
  a21=(a10*a21);
  if (res[1]!=0) res[1][28]=a21;
  a20=(a17*a20);
  a16=(a18*a16);
  a20=(a20-a16);
  a19=(a13*a19);
  a20=(a20-a19);
  a14=(a15*a14);
  a20=(a20+a14);
  a20=(a10*a20);
  if (res[1]!=0) res[1][29]=a20;
  a20=arg[1]? arg[1][37] : 0;
  if (res[1]!=0) res[1][30]=a20;
  a20=arg[1]? arg[1][38] : 0;
  if (res[1]!=0) res[1][31]=a20;
  a20=arg[1]? arg[1][39] : 0;
  if (res[1]!=0) res[1][32]=a20;
  a20=arg[1]? arg[1][34] : 0;
  a14=(a1*a20);
  a19=arg[1]? arg[1][35] : 0;
  a16=(a4*a19);
  a14=(a14+a16);
  a16=arg[1]? arg[1][36] : 0;
  a21=(a6*a16);
  a14=(a14+a21);
  a14=(a0*a14);
  a14=(-a14);
  if (res[1]!=0) res[1][33]=a14;
  a14=arg[1]? arg[1][33] : 0;
  a21=(a1*a14);
  a22=(a6*a19);
  a21=(a21+a22);
  a22=(a4*a16);
  a21=(a21-a22);
  a21=(a0*a21);
  if (res[1]!=0) res[1][34]=a21;
  a21=(a4*a14);
  a22=(a6*a20);
  a21=(a21-a22);
  a22=(a1*a16);
  a21=(a21+a22);
  a21=(a0*a21);
  if (res[1]!=0) res[1][35]=a21;
  a21=(a6*a14);
  a22=(a4*a20);
  a21=(a21+a22);
  a22=(a1*a19);
  a21=(a21-a22);
  a21=(a0*a21);
  if (res[1]!=0) res[1][36]=a21;
  a21=(a5*a14);
  a22=(a3*a19);
  a21=(a21+a22);
  a22=(a7*a20);
  a23=(a2*a16);
  a22=(a22+a23);
  a21=(a21+a22);
  a21=(a8*a21);
  a21=(a10*a21);
  if (res[1]!=0) res[1][37]=a21;
  a21=(a7*a19);
  a22=(a5*a16);
  a21=(a21+a22);
  a22=(a2*a14);
  a23=(a3*a20);
  a22=(a22+a23);
  a21=(a21-a22);
  a21=(a8*a21);
  a21=(a10*a21);
  if (res[1]!=0) res[1][38]=a21;
  a14=(a17*a14);
  a20=(a18*a20);
  a14=(a14-a20);
  a19=(a13*a19);
  a14=(a14-a19);
  a16=(a15*a16);
  a14=(a14+a16);
  a14=(a10*a14);
  if (res[1]!=0) res[1][39]=a14;
  a14=arg[1]? arg[1][47] : 0;
  if (res[1]!=0) res[1][40]=a14;
  a14=arg[1]? arg[1][48] : 0;
  if (res[1]!=0) res[1][41]=a14;
  a14=arg[1]? arg[1][49] : 0;
  if (res[1]!=0) res[1][42]=a14;
  a14=arg[1]? arg[1][44] : 0;
  a16=(a1*a14);
  a19=arg[1]? arg[1][45] : 0;
  a20=(a4*a19);
  a16=(a16+a20);
  a20=arg[1]? arg[1][46] : 0;
  a21=(a6*a20);
  a16=(a16+a21);
  a16=(a0*a16);
  a16=(-a16);
  if (res[1]!=0) res[1][43]=a16;
  a16=arg[1]? arg[1][43] : 0;
  a21=(a1*a16);
  a22=(a6*a19);
  a21=(a21+a22);
  a22=(a4*a20);
  a21=(a21-a22);
  a21=(a0*a21);
  if (res[1]!=0) res[1][44]=a21;
  a21=(a4*a16);
  a22=(a6*a14);
  a21=(a21-a22);
  a22=(a1*a20);
  a21=(a21+a22);
  a21=(a0*a21);
  if (res[1]!=0) res[1][45]=a21;
  a21=(a6*a16);
  a22=(a4*a14);
  a21=(a21+a22);
  a22=(a1*a19);
  a21=(a21-a22);
  a21=(a0*a21);
  if (res[1]!=0) res[1][46]=a21;
  a21=(a5*a16);
  a22=(a3*a19);
  a21=(a21+a22);
  a22=(a7*a14);
  a23=(a2*a20);
  a22=(a22+a23);
  a21=(a21+a22);
  a21=(a8*a21);
  a21=(a10*a21);
  if (res[1]!=0) res[1][47]=a21;
  a21=(a7*a19);
  a22=(a5*a20);
  a21=(a21+a22);
  a22=(a2*a16);
  a23=(a3*a14);
  a22=(a22+a23);
  a21=(a21-a22);
  a21=(a8*a21);
  a21=(a10*a21);
  if (res[1]!=0) res[1][48]=a21;
  a16=(a17*a16);
  a14=(a18*a14);
  a16=(a16-a14);
  a19=(a13*a19);
  a16=(a16-a19);
  a20=(a15*a20);
  a16=(a16+a20);
  a16=(a10*a16);
  if (res[1]!=0) res[1][49]=a16;
  a16=arg[1]? arg[1][57] : 0;
  if (res[1]!=0) res[1][50]=a16;
  a16=arg[1]? arg[1][58] : 0;
  if (res[1]!=0) res[1][51]=a16;
  a16=arg[1]? arg[1][59] : 0;
  if (res[1]!=0) res[1][52]=a16;
  a16=arg[1]? arg[1][54] : 0;
  a20=(a1*a16);
  a19=arg[1]? arg[1][55] : 0;
  a14=(a4*a19);
  a20=(a20+a14);
  a14=arg[1]? arg[1][56] : 0;
  a21=(a6*a14);
  a20=(a20+a21);
  a20=(a0*a20);
  a20=(-a20);
  if (res[1]!=0) res[1][53]=a20;
  a20=arg[1]? arg[1][53] : 0;
  a21=(a1*a20);
  a22=(a6*a19);
  a21=(a21+a22);
  a22=(a4*a14);
  a21=(a21-a22);
  a21=(a0*a21);
  if (res[1]!=0) res[1][54]=a21;
  a21=(a4*a20);
  a22=(a6*a16);
  a21=(a21-a22);
  a22=(a1*a14);
  a21=(a21+a22);
  a21=(a0*a21);
  if (res[1]!=0) res[1][55]=a21;
  a21=(a6*a20);
  a22=(a4*a16);
  a21=(a21+a22);
  a22=(a1*a19);
  a21=(a21-a22);
  a21=(a0*a21);
  if (res[1]!=0) res[1][56]=a21;
  a21=(a5*a20);
  a22=(a3*a19);
  a21=(a21+a22);
  a22=(a7*a16);
  a23=(a2*a14);
  a22=(a22+a23);
  a21=(a21+a22);
  a21=(a8*a21);
  a21=(a10*a21);
  if (res[1]!=0) res[1][57]=a21;
  a21=(a7*a19);
  a22=(a5*a14);
  a21=(a21+a22);
  a22=(a2*a20);
  a23=(a3*a16);
  a22=(a22+a23);
  a21=(a21-a22);
  a21=(a8*a21);
  a21=(a10*a21);
  if (res[1]!=0) res[1][58]=a21;
  a20=(a17*a20);
  a16=(a18*a16);
  a20=(a20-a16);
  a19=(a13*a19);
  a20=(a20-a19);
  a14=(a15*a14);
  a20=(a20+a14);
  a20=(a10*a20);
  if (res[1]!=0) res[1][59]=a20;
  a20=arg[1]? arg[1][67] : 0;
  if (res[1]!=0) res[1][60]=a20;
  a20=arg[1]? arg[1][68] : 0;
  if (res[1]!=0) res[1][61]=a20;
  a20=arg[1]? arg[1][69] : 0;
  if (res[1]!=0) res[1][62]=a20;
  a20=arg[1]? arg[1][64] : 0;
  a14=(a1*a20);
  a19=arg[1]? arg[1][65] : 0;
  a16=(a4*a19);
  a14=(a14+a16);
  a16=arg[1]? arg[1][66] : 0;
  a21=(a6*a16);
  a14=(a14+a21);
  a14=(a0*a14);
  a14=(-a14);
  if (res[1]!=0) res[1][63]=a14;
  a14=arg[1]? arg[1][63] : 0;
  a21=(a1*a14);
  a22=(a6*a19);
  a21=(a21+a22);
  a22=(a4*a16);
  a21=(a21-a22);
  a21=(a0*a21);
  if (res[1]!=0) res[1][64]=a21;
  a21=(a4*a14);
  a22=(a6*a20);
  a21=(a21-a22);
  a22=(a1*a16);
  a21=(a21+a22);
  a21=(a0*a21);
  if (res[1]!=0) res[1][65]=a21;
  a21=(a6*a14);
  a22=(a4*a20);
  a21=(a21+a22);
  a22=(a1*a19);
  a21=(a21-a22);
  a21=(a0*a21);
  if (res[1]!=0) res[1][66]=a21;
  a21=(a5*a14);
  a22=(a3*a19);
  a21=(a21+a22);
  a22=(a7*a20);
  a23=(a2*a16);
  a22=(a22+a23);
  a21=(a21+a22);
  a21=(a8*a21);
  a21=(a10*a21);
  if (res[1]!=0) res[1][67]=a21;
  a21=(a7*a19);
  a22=(a5*a16);
  a21=(a21+a22);
  a22=(a2*a14);
  a23=(a3*a20);
  a22=(a22+a23);
  a21=(a21-a22);
  a21=(a8*a21);
  a21=(a10*a21);
  if (res[1]!=0) res[1][68]=a21;
  a14=(a17*a14);
  a20=(a18*a20);
  a14=(a14-a20);
  a19=(a13*a19);
  a14=(a14-a19);
  a16=(a15*a16);
  a14=(a14+a16);
  a14=(a10*a14);
  if (res[1]!=0) res[1][69]=a14;
  a14=arg[1]? arg[1][77] : 0;
  if (res[1]!=0) res[1][70]=a14;
  a14=arg[1]? arg[1][78] : 0;
  if (res[1]!=0) res[1][71]=a14;
  a14=arg[1]? arg[1][79] : 0;
  if (res[1]!=0) res[1][72]=a14;
  a14=arg[1]? arg[1][74] : 0;
  a16=(a1*a14);
  a19=arg[1]? arg[1][75] : 0;
  a20=(a4*a19);
  a16=(a16+a20);
  a20=arg[1]? arg[1][76] : 0;
  a21=(a6*a20);
  a16=(a16+a21);
  a16=(a0*a16);
  a16=(-a16);
  if (res[1]!=0) res[1][73]=a16;
  a16=arg[1]? arg[1][73] : 0;
  a21=(a1*a16);
  a22=(a6*a19);
  a21=(a21+a22);
  a22=(a4*a20);
  a21=(a21-a22);
  a21=(a0*a21);
  if (res[1]!=0) res[1][74]=a21;
  a21=(a4*a16);
  a22=(a6*a14);
  a21=(a21-a22);
  a22=(a1*a20);
  a21=(a21+a22);
  a21=(a0*a21);
  if (res[1]!=0) res[1][75]=a21;
  a21=(a6*a16);
  a22=(a4*a14);
  a21=(a21+a22);
  a22=(a1*a19);
  a21=(a21-a22);
  a21=(a0*a21);
  if (res[1]!=0) res[1][76]=a21;
  a21=(a5*a16);
  a22=(a3*a19);
  a21=(a21+a22);
  a22=(a7*a14);
  a23=(a2*a20);
  a22=(a22+a23);
  a21=(a21+a22);
  a21=(a8*a21);
  a21=(a10*a21);
  if (res[1]!=0) res[1][77]=a21;
  a21=(a7*a19);
  a22=(a5*a20);
  a21=(a21+a22);
  a22=(a2*a16);
  a23=(a3*a14);
  a22=(a22+a23);
  a21=(a21-a22);
  a21=(a8*a21);
  a21=(a10*a21);
  if (res[1]!=0) res[1][78]=a21;
  a16=(a17*a16);
  a14=(a18*a14);
  a16=(a16-a14);
  a19=(a13*a19);
  a16=(a16-a19);
  a20=(a15*a20);
  a16=(a16+a20);
  a16=(a10*a16);
  if (res[1]!=0) res[1][79]=a16;
  a16=arg[1]? arg[1][87] : 0;
  if (res[1]!=0) res[1][80]=a16;
  a16=arg[1]? arg[1][88] : 0;
  if (res[1]!=0) res[1][81]=a16;
  a16=arg[1]? arg[1][89] : 0;
  if (res[1]!=0) res[1][82]=a16;
  a16=arg[1]? arg[1][84] : 0;
  a20=(a1*a16);
  a19=arg[1]? arg[1][85] : 0;
  a14=(a4*a19);
  a20=(a20+a14);
  a14=arg[1]? arg[1][86] : 0;
  a21=(a6*a14);
  a20=(a20+a21);
  a20=(a0*a20);
  a20=(-a20);
  if (res[1]!=0) res[1][83]=a20;
  a20=arg[1]? arg[1][83] : 0;
  a21=(a1*a20);
  a22=(a6*a19);
  a21=(a21+a22);
  a22=(a4*a14);
  a21=(a21-a22);
  a21=(a0*a21);
  if (res[1]!=0) res[1][84]=a21;
  a21=(a4*a20);
  a22=(a6*a16);
  a21=(a21-a22);
  a22=(a1*a14);
  a21=(a21+a22);
  a21=(a0*a21);
  if (res[1]!=0) res[1][85]=a21;
  a21=(a6*a20);
  a22=(a4*a16);
  a21=(a21+a22);
  a22=(a1*a19);
  a21=(a21-a22);
  a21=(a0*a21);
  if (res[1]!=0) res[1][86]=a21;
  a21=(a5*a20);
  a22=(a3*a19);
  a21=(a21+a22);
  a22=(a7*a16);
  a23=(a2*a14);
  a22=(a22+a23);
  a21=(a21+a22);
  a21=(a8*a21);
  a21=(a10*a21);
  if (res[1]!=0) res[1][87]=a21;
  a21=(a7*a19);
  a22=(a5*a14);
  a21=(a21+a22);
  a22=(a2*a20);
  a23=(a3*a16);
  a22=(a22+a23);
  a21=(a21-a22);
  a21=(a8*a21);
  a21=(a10*a21);
  if (res[1]!=0) res[1][88]=a21;
  a20=(a17*a20);
  a16=(a18*a16);
  a20=(a20-a16);
  a19=(a13*a19);
  a20=(a20-a19);
  a14=(a15*a14);
  a20=(a20+a14);
  a20=(a10*a20);
  if (res[1]!=0) res[1][89]=a20;
  a20=arg[1]? arg[1][97] : 0;
  if (res[1]!=0) res[1][90]=a20;
  a20=arg[1]? arg[1][98] : 0;
  if (res[1]!=0) res[1][91]=a20;
  a20=arg[1]? arg[1][99] : 0;
  if (res[1]!=0) res[1][92]=a20;
  a20=arg[1]? arg[1][94] : 0;
  a14=(a1*a20);
  a19=arg[1]? arg[1][95] : 0;
  a16=(a4*a19);
  a14=(a14+a16);
  a16=arg[1]? arg[1][96] : 0;
  a21=(a6*a16);
  a14=(a14+a21);
  a14=(a0*a14);
  a14=(-a14);
  if (res[1]!=0) res[1][93]=a14;
  a14=arg[1]? arg[1][93] : 0;
  a21=(a1*a14);
  a22=(a6*a19);
  a21=(a21+a22);
  a22=(a4*a16);
  a21=(a21-a22);
  a21=(a0*a21);
  if (res[1]!=0) res[1][94]=a21;
  a21=(a4*a14);
  a22=(a6*a20);
  a21=(a21-a22);
  a22=(a1*a16);
  a21=(a21+a22);
  a21=(a0*a21);
  if (res[1]!=0) res[1][95]=a21;
  a21=(a6*a14);
  a22=(a4*a20);
  a21=(a21+a22);
  a22=(a1*a19);
  a21=(a21-a22);
  a21=(a0*a21);
  if (res[1]!=0) res[1][96]=a21;
  a21=(a5*a14);
  a22=(a3*a19);
  a21=(a21+a22);
  a22=(a7*a20);
  a23=(a2*a16);
  a22=(a22+a23);
  a21=(a21+a22);
  a21=(a8*a21);
  a21=(a10*a21);
  if (res[1]!=0) res[1][97]=a21;
  a21=(a7*a19);
  a22=(a5*a16);
  a21=(a21+a22);
  a22=(a2*a14);
  a23=(a3*a20);
  a22=(a22+a23);
  a21=(a21-a22);
  a21=(a8*a21);
  a21=(a10*a21);
  if (res[1]!=0) res[1][98]=a21;
  a17=(a17*a14);
  a18=(a18*a20);
  a17=(a17-a18);
  a13=(a13*a19);
  a17=(a17-a13);
  a15=(a15*a16);
  a17=(a17+a15);
  a17=(a10*a17);
  if (res[1]!=0) res[1][99]=a17;
  a17=arg[2]? arg[2][7] : 0;
  if (res[2]!=0) res[2][0]=a17;
  a17=arg[2]? arg[2][8] : 0;
  if (res[2]!=0) res[2][1]=a17;
  a17=arg[2]? arg[2][9] : 0;
  if (res[2]!=0) res[2][2]=a17;
  a17=(a0*a2);
  a15=arg[2]? arg[2][4] : 0;
  a16=(a1*a15);
  a13=arg[2]? arg[2][5] : 0;
  a19=(a4*a13);
  a16=(a16+a19);
  a19=arg[2]? arg[2][6] : 0;
  a18=(a6*a19);
  a16=(a16+a18);
  a16=(a0*a16);
  a17=(a17+a16);
  a17=(-a17);
  if (res[2]!=0) res[2][3]=a17;
  a17=(a0*a3);
  a16=arg[2]? arg[2][3] : 0;
  a18=(a1*a16);
  a20=(a6*a13);
  a18=(a18+a20);
  a20=(a4*a19);
  a18=(a18-a20);
  a18=(a0*a18);
  a17=(a17+a18);
  if (res[2]!=0) res[2][4]=a17;
  a17=(a0*a7);
  a18=(a4*a16);
  a20=(a6*a15);
  a18=(a18-a20);
  a20=(a1*a19);
  a18=(a18+a20);
  a18=(a0*a18);
  a17=(a17+a18);
  if (res[2]!=0) res[2][5]=a17;
  a17=(a6*a16);
  a18=(a4*a15);
  a17=(a17+a18);
  a18=(a1*a13);
  a17=(a17-a18);
  a17=(a0*a17);
  a18=(a0*a5);
  a17=(a17-a18);
  if (res[2]!=0) res[2][6]=a17;
  a17=(a5*a16);
  a18=(a3*a13);
  a17=(a17+a18);
  a18=(a7*a15);
  a20=(a2*a19);
  a18=(a18+a20);
  a17=(a17+a18);
  a17=(a8*a17);
  a17=(a10*a17);
  if (res[2]!=0) res[2][7]=a17;
  a17=(a7*a13);
  a18=(a5*a19);
  a17=(a17+a18);
  a18=(a2*a16);
  a20=(a3*a15);
  a18=(a18+a20);
  a17=(a17-a18);
  a17=(a8*a17);
  a17=(a10*a17);
  if (res[2]!=0) res[2][8]=a17;
  a17=(a3+a3);
  a16=(a17*a16);
  a18=(a2+a2);
  a15=(a18*a15);
  a16=(a16-a15);
  a15=(a5+a5);
  a13=(a15*a13);
  a16=(a16-a13);
  a13=(a7+a7);
  a19=(a13*a19);
  a16=(a16+a19);
  a16=(a10*a16);
  if (res[2]!=0) res[2][9]=a16;
  a16=arg[2]? arg[2][17] : 0;
  if (res[2]!=0) res[2][10]=a16;
  a16=arg[2]? arg[2][18] : 0;
  if (res[2]!=0) res[2][11]=a16;
  a16=arg[2]? arg[2][19] : 0;
  if (res[2]!=0) res[2][12]=a16;
  a16=(a0*a5);
  a19=arg[2]? arg[2][14] : 0;
  a20=(a1*a19);
  a14=arg[2]? arg[2][15] : 0;
  a21=(a4*a14);
  a20=(a20+a21);
  a21=arg[2]? arg[2][16] : 0;
  a22=(a6*a21);
  a20=(a20+a22);
  a20=(a0*a20);
  a16=(a16+a20);
  a16=(-a16);
  if (res[2]!=0) res[2][13]=a16;
  a16=arg[2]? arg[2][13] : 0;
  a20=(a1*a16);
  a22=(a6*a14);
  a20=(a20+a22);
  a22=(a4*a21);
  a20=(a20-a22);
  a20=(a0*a20);
  a22=(a0*a7);
  a20=(a20-a22);
  if (res[2]!=0) res[2][14]=a20;
  a20=(a0*a3);
  a22=(a4*a16);
  a23=(a6*a19);
  a22=(a22-a23);
  a23=(a1*a21);
  a22=(a22+a23);
  a22=(a0*a22);
  a20=(a20+a22);
  if (res[2]!=0) res[2][15]=a20;
  a20=(a0*a2);
  a22=(a6*a16);
  a23=(a4*a19);
  a22=(a22+a23);
  a23=(a1*a14);
  a22=(a22-a23);
  a22=(a0*a22);
  a20=(a20+a22);
  if (res[2]!=0) res[2][16]=a20;
  a20=(a5*a16);
  a22=(a3*a14);
  a20=(a20+a22);
  a22=(a7*a19);
  a23=(a2*a21);
  a22=(a22+a23);
  a20=(a20+a22);
  a20=(a8*a20);
  a20=(a10*a20);
  if (res[2]!=0) res[2][17]=a20;
  a20=(a7*a14);
  a22=(a5*a21);
  a20=(a20+a22);
  a22=(a2*a16);
  a23=(a3*a19);
  a22=(a22+a23);
  a20=(a20-a22);
  a20=(a8*a20);
  a20=(a10*a20);
  if (res[2]!=0) res[2][18]=a20;
  a16=(a17*a16);
  a19=(a18*a19);
  a16=(a16-a19);
  a14=(a15*a14);
  a16=(a16-a14);
  a21=(a13*a21);
  a16=(a16+a21);
  a16=(a10*a16);
  if (res[2]!=0) res[2][19]=a16;
  a16=arg[2]? arg[2][27] : 0;
  if (res[2]!=0) res[2][20]=a16;
  a16=arg[2]? arg[2][28] : 0;
  if (res[2]!=0) res[2][21]=a16;
  a16=arg[2]? arg[2][29] : 0;
  if (res[2]!=0) res[2][22]=a16;
  a16=(a0*a7);
  a21=arg[2]? arg[2][24] : 0;
  a14=(a1*a21);
  a19=arg[2]? arg[2][25] : 0;
  a20=(a4*a19);
  a14=(a14+a20);
  a20=arg[2]? arg[2][26] : 0;
  a22=(a6*a20);
  a14=(a14+a22);
  a14=(a0*a14);
  a16=(a16+a14);
  a16=(-a16);
  if (res[2]!=0) res[2][23]=a16;
  a16=(a0*a5);
  a14=arg[2]? arg[2][23] : 0;
  a22=(a1*a14);
  a23=(a6*a19);
  a22=(a22+a23);
  a23=(a4*a20);
  a22=(a22-a23);
  a22=(a0*a22);
  a16=(a16+a22);
  if (res[2]!=0) res[2][24]=a16;
  a16=(a4*a14);
  a22=(a6*a21);
  a16=(a16-a22);
  a22=(a1*a20);
  a16=(a16+a22);
  a16=(a0*a16);
  a22=(a0*a2);
  a16=(a16-a22);
  if (res[2]!=0) res[2][25]=a16;
  a16=(a0*a3);
  a22=(a6*a14);
  a23=(a4*a21);
  a22=(a22+a23);
  a23=(a1*a19);
  a22=(a22-a23);
  a22=(a0*a22);
  a16=(a16+a22);
  if (res[2]!=0) res[2][26]=a16;
  a16=(a5*a14);
  a22=(a3*a19);
  a16=(a16+a22);
  a22=(a7*a21);
  a23=(a2*a20);
  a22=(a22+a23);
  a16=(a16+a22);
  a16=(a8*a16);
  a16=(a10*a16);
  if (res[2]!=0) res[2][27]=a16;
  a16=(a7*a19);
  a22=(a5*a20);
  a16=(a16+a22);
  a22=(a2*a14);
  a23=(a3*a21);
  a22=(a22+a23);
  a16=(a16-a22);
  a16=(a8*a16);
  a16=(a10*a16);
  if (res[2]!=0) res[2][28]=a16;
  a14=(a17*a14);
  a21=(a18*a21);
  a14=(a14-a21);
  a19=(a15*a19);
  a14=(a14-a19);
  a20=(a13*a20);
  a14=(a14+a20);
  a14=(a10*a14);
  if (res[2]!=0) res[2][29]=a14;
  a14=arg[2]? arg[2][37] : 0;
  if (res[2]!=0) res[2][30]=a14;
  a14=arg[2]? arg[2][38] : 0;
  if (res[2]!=0) res[2][31]=a14;
  a14=arg[2]? arg[2][39] : 0;
  if (res[2]!=0) res[2][32]=a14;
  a14=arg[2]? arg[2][34] : 0;
  a20=(a1*a14);
  a19=arg[2]? arg[2][35] : 0;
  a21=(a4*a19);
  a20=(a20+a21);
  a21=arg[2]? arg[2][36] : 0;
  a16=(a6*a21);
  a20=(a20+a16);
  a20=(a0*a20);
  a20=(-a20);
  if (res[2]!=0) res[2][33]=a20;
  a20=arg[2]? arg[2][33] : 0;
  a16=(a1*a20);
  a22=(a6*a19);
  a16=(a16+a22);
  a22=(a4*a21);
  a16=(a16-a22);
  a16=(a0*a16);
  if (res[2]!=0) res[2][34]=a16;
  a16=(a4*a20);
  a22=(a6*a14);
  a16=(a16-a22);
  a22=(a1*a21);
  a16=(a16+a22);
  a16=(a0*a16);
  if (res[2]!=0) res[2][35]=a16;
  a6=(a6*a20);
  a4=(a4*a14);
  a6=(a6+a4);
  a1=(a1*a19);
  a6=(a6-a1);
  a0=(a0*a6);
  if (res[2]!=0) res[2][36]=a0;
  a0=(a5*a20);
  a6=(a3*a19);
  a0=(a0+a6);
  a6=(a7*a14);
  a1=(a2*a21);
  a6=(a6+a1);
  a0=(a0+a6);
  a0=(a8*a0);
  a0=(a10*a0);
  a9=(a9+a0);
  if (res[2]!=0) res[2][37]=a9;
  a7=(a7*a19);
  a5=(a5*a21);
  a7=(a7+a5);
  a2=(a2*a20);
  a3=(a3*a14);
  a2=(a2+a3);
  a7=(a7-a2);
  a8=(a8*a7);
  a8=(a10*a8);
  a11=(a11+a8);
  if (res[2]!=0) res[2][38]=a11;
  a17=(a17*a20);
  a18=(a18*a14);
  a17=(a17-a18);
  a15=(a15*a19);
  a17=(a17-a15);
  a13=(a13*a21);
  a17=(a17+a13);
  a10=(a10*a17);
  a12=(a12+a10);
  if (res[2]!=0) res[2][39]=a12;
  return 0;
}

CASADI_SYMBOL_EXPORT int quadrotor_q_p_expl_vde_forw(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int quadrotor_q_p_expl_vde_forw_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int quadrotor_q_p_expl_vde_forw_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void quadrotor_q_p_expl_vde_forw_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int quadrotor_q_p_expl_vde_forw_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void quadrotor_q_p_expl_vde_forw_release(int mem) {
}

CASADI_SYMBOL_EXPORT void quadrotor_q_p_expl_vde_forw_incref(void) {
}

CASADI_SYMBOL_EXPORT void quadrotor_q_p_expl_vde_forw_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int quadrotor_q_p_expl_vde_forw_n_in(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_int quadrotor_q_p_expl_vde_forw_n_out(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_real quadrotor_q_p_expl_vde_forw_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* quadrotor_q_p_expl_vde_forw_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* quadrotor_q_p_expl_vde_forw_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* quadrotor_q_p_expl_vde_forw_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    case 3: return casadi_s3;
    case 4: return casadi_s4;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* quadrotor_q_p_expl_vde_forw_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int quadrotor_q_p_expl_vde_forw_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 5;
  if (sz_res) *sz_res = 3;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
