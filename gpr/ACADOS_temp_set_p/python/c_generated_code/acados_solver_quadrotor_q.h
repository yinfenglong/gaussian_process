/*
 * Copyright 2019 Gianluca Frison, Dimitris Kouzoupis, Robin Verschueren,
 * Andrea Zanelli, Niels van Duijkeren, Jonathan Frey, Tommaso Sartor,
 * Branimir Novoselnik, Rien Quirynen, Rezart Qelibari, Dang Doan,
 * Jonas Koenemann, Yutao Chen, Tobias Schöls, Jonas Schlagenhauf, Moritz Diehl
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */

#ifndef ACADOS_SOLVER_quadrotor_q_H_
#define ACADOS_SOLVER_quadrotor_q_H_

#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

#define QUADROTOR_Q_NX     10
#define QUADROTOR_Q_NZ     0
#define QUADROTOR_Q_NU     4
#define QUADROTOR_Q_NP     0
#define QUADROTOR_Q_NBX    0
#define QUADROTOR_Q_NBX0   10
#define QUADROTOR_Q_NBU    4
#define QUADROTOR_Q_NSBX   0
#define QUADROTOR_Q_NSBU   0
#define QUADROTOR_Q_NSH    0
#define QUADROTOR_Q_NSG    0
#define QUADROTOR_Q_NSPHI  0
#define QUADROTOR_Q_NSHN   0
#define QUADROTOR_Q_NSGN   0
#define QUADROTOR_Q_NSPHIN 0
#define QUADROTOR_Q_NSBXN  0
#define QUADROTOR_Q_NS     0
#define QUADROTOR_Q_NSN    0
#define QUADROTOR_Q_NG     0
#define QUADROTOR_Q_NBXN   0
#define QUADROTOR_Q_NGN    0
#define QUADROTOR_Q_NY0    14
#define QUADROTOR_Q_NY     14
#define QUADROTOR_Q_NYN    6
#define QUADROTOR_Q_N      20
#define QUADROTOR_Q_NH     0
#define QUADROTOR_Q_NPHI   0
#define QUADROTOR_Q_NHN    0
#define QUADROTOR_Q_NPHIN  0
#define QUADROTOR_Q_NR     0

#ifdef __cplusplus
extern "C" {
#endif

// ** capsule for solver data **
typedef struct quadrotor_q_solver_capsule
{
    // acados objects
    ocp_nlp_in *nlp_in;
    ocp_nlp_out *nlp_out;
    ocp_nlp_solver *nlp_solver;
    void *nlp_opts;
    ocp_nlp_plan *nlp_solver_plan;
    ocp_nlp_config *nlp_config;
    ocp_nlp_dims *nlp_dims;

    // number of expected runtime parameters
    unsigned int nlp_np;

    /* external functions */
    // dynamics

    external_function_param_casadi *forw_vde_casadi;
    external_function_param_casadi *expl_ode_fun;




    // cost






    // constraints




} quadrotor_q_solver_capsule;

quadrotor_q_solver_capsule * quadrotor_q_acados_create_capsule(void);
int quadrotor_q_acados_free_capsule(quadrotor_q_solver_capsule *capsule);

int quadrotor_q_acados_create(quadrotor_q_solver_capsule * capsule);
int quadrotor_q_acados_update_params(quadrotor_q_solver_capsule * capsule, int stage, double *value, int np);
int quadrotor_q_acados_solve(quadrotor_q_solver_capsule * capsule);
int quadrotor_q_acados_free(quadrotor_q_solver_capsule * capsule);
void quadrotor_q_acados_print_stats(quadrotor_q_solver_capsule * capsule);

ocp_nlp_in *quadrotor_q_acados_get_nlp_in(quadrotor_q_solver_capsule * capsule);
ocp_nlp_out *quadrotor_q_acados_get_nlp_out(quadrotor_q_solver_capsule * capsule);
ocp_nlp_solver *quadrotor_q_acados_get_nlp_solver(quadrotor_q_solver_capsule * capsule);
ocp_nlp_config *quadrotor_q_acados_get_nlp_config(quadrotor_q_solver_capsule * capsule);
void *quadrotor_q_acados_get_nlp_opts(quadrotor_q_solver_capsule * capsule);
ocp_nlp_dims *quadrotor_q_acados_get_nlp_dims(quadrotor_q_solver_capsule * capsule);
ocp_nlp_plan *quadrotor_q_acados_get_nlp_plan(quadrotor_q_solver_capsule * capsule);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_SOLVER_quadrotor_q_H_
