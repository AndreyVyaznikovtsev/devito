/* Devito generated code for Operator `ForwardSH` */

#define _POSIX_C_SOURCE 200809L
#define START(S) struct timeval start_ ## S , end_ ## S ; gettimeofday(&start_ ## S , NULL);
#define STOP(S,T) gettimeofday(&end_ ## S, NULL); T->S += (double)(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;

#include <cstdlib>
#include <cmath>
#include "sys/time.h"
#include "openacc.h"

struct dataobj
{
  void *__restrict data;
  int * size;
  unsigned long nbytes;
  unsigned long * npsize;
  unsigned long * dsize;
  int * hsize;
  int * hofs;
  int * oofs;
  void * dmap;
} ;

struct profiler
{
  double section0;
  double section1;
  double section2;
} ;

extern "C" int ForwardSH(struct dataobj *__restrict b_vec, struct dataobj *__restrict damp_vec, struct dataobj *__restrict mu_x_vec, struct dataobj *__restrict mu_z_vec, struct dataobj *__restrict rec_vec, struct dataobj *__restrict rec_coords_vec, struct dataobj *__restrict src_vec, struct dataobj *__restrict src_coords_vec, struct dataobj *__restrict tau_xy_vec, struct dataobj *__restrict tau_zy_vec, struct dataobj *__restrict v_vec, const int x_M, const int x_m, const int y_M, const int y_m, const float dt, const float o_x, const float o_y, const int p_rec_M, const int p_rec_m, const int p_src_M, const int p_src_m, const int time_M, const int time_m, const int deviceid, const int devicerm, struct profiler * timers);


int ForwardSH(struct dataobj *__restrict b_vec, struct dataobj *__restrict damp_vec, struct dataobj *__restrict mu_x_vec, struct dataobj *__restrict mu_z_vec, struct dataobj *__restrict rec_vec, struct dataobj *__restrict rec_coords_vec, struct dataobj *__restrict src_vec, struct dataobj *__restrict src_coords_vec, struct dataobj *__restrict tau_xy_vec, struct dataobj *__restrict tau_zy_vec, struct dataobj *__restrict v_vec, const int x_M, const int x_m, const int y_M, const int y_m, const float dt, const float o_x, const float o_y, const int p_rec_M, const int p_rec_m, const int p_src_M, const int p_src_m, const int time_M, const int time_m, const int deviceid, const int devicerm, struct profiler * timers)
{
  /* Beginning of OpenACC setup */
  acc_init(acc_device_nvidia);
  if (deviceid != -1)
  {
    acc_set_device_num(deviceid,acc_device_nvidia);
  }
  /* End of OpenACC setup */

  float (*__restrict b)[b_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[b_vec->size[1]]) b_vec->data;
  float (*__restrict damp)[damp_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[damp_vec->size[1]]) damp_vec->data;
  float (*__restrict mu_x)[mu_x_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[mu_x_vec->size[1]]) mu_x_vec->data;
  float (*__restrict mu_z)[mu_z_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[mu_z_vec->size[1]]) mu_z_vec->data;
  float (*__restrict rec)[rec_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec_vec->size[1]]) rec_vec->data;
  float (*__restrict rec_coords)[rec_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec_coords_vec->size[1]]) rec_coords_vec->data;
  float (*__restrict src)[src_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[src_vec->size[1]]) src_vec->data;
  float (*__restrict src_coords)[src_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[src_coords_vec->size[1]]) src_coords_vec->data;
  float (*__restrict tau_xy)[tau_xy_vec->size[1]][tau_xy_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[tau_xy_vec->size[1]][tau_xy_vec->size[2]]) tau_xy_vec->data;
  float (*__restrict tau_zy)[tau_zy_vec->size[1]][tau_zy_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[tau_zy_vec->size[1]][tau_zy_vec->size[2]]) tau_zy_vec->data;
  float (*__restrict v)[v_vec->size[1]][v_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[v_vec->size[1]][v_vec->size[2]]) v_vec->data;

  #pragma acc enter data copyin(rec[0:rec_vec->size[0]][0:rec_vec->size[1]])
  #pragma acc enter data copyin(tau_xy[0:tau_xy_vec->size[0]][0:tau_xy_vec->size[1]][0:tau_xy_vec->size[2]])
  #pragma acc enter data copyin(tau_zy[0:tau_zy_vec->size[0]][0:tau_zy_vec->size[1]][0:tau_zy_vec->size[2]])
  #pragma acc enter data copyin(v[0:v_vec->size[0]][0:v_vec->size[1]][0:v_vec->size[2]])
  #pragma acc enter data copyin(b[0:b_vec->size[0]][0:b_vec->size[1]])
  #pragma acc enter data copyin(damp[0:damp_vec->size[0]][0:damp_vec->size[1]])
  #pragma acc enter data copyin(mu_x[0:mu_x_vec->size[0]][0:mu_x_vec->size[1]])
  #pragma acc enter data copyin(mu_z[0:mu_z_vec->size[0]][0:mu_z_vec->size[1]])
  #pragma acc enter data copyin(rec_coords[0:rec_coords_vec->size[0]][0:rec_coords_vec->size[1]])
  #pragma acc enter data copyin(src[0:src_vec->size[0]][0:src_vec->size[1]])
  #pragma acc enter data copyin(src_coords[0:src_coords_vec->size[0]][0:src_coords_vec->size[1]])

  const float r1 = 1.0F/dt;
  for (int time = time_m, t0 = (time)%(2), t1 = (time + 1)%(2); time <= time_M; time += 1, t0 = (time)%(2), t1 = (time + 1)%(2))
  {
    START(section0)
    #pragma acc parallel loop collapse(2) present(b,damp,tau_xy,tau_zy,v)
    for (int x = x_m; x <= x_M; x += 1)
    {
      for (int y = y_m; y <= y_M; y += 1)
      {
        v[t1][x + 4][y + 4] = dt*(r1*v[t0][x + 4][y + 4] + (1.04166667e-1F*(tau_xy[t0][x + 2][y + 4] - tau_xy[t0][x + 5][y + 4] + tau_zy[t0][x + 4][y + 2] - tau_zy[t0][x + 4][y + 5]) + 2.81250F*(-tau_xy[t0][x + 3][y + 4] + tau_xy[t0][x + 4][y + 4] - tau_zy[t0][x + 4][y + 3] + tau_zy[t0][x + 4][y + 4]))*b[x + 4][y + 4])*damp[x + 4][y + 4];
      }
    }
    #pragma acc parallel loop collapse(2) present(damp,mu_x,mu_z,tau_xy,tau_zy,v)
    for (int x = x_m; x <= x_M; x += 1)
    {
      for (int y = y_m; y <= y_M; y += 1)
      {
        tau_xy[t1][x + 4][y + 4] = 5.0e-1F*dt*(r1*tau_xy[t0][x + 4][y + 4] + (1.04166667e-1F*(v[t1][x + 3][y + 4] - v[t1][x + 6][y + 4]) + 2.81250F*(-v[t1][x + 4][y + 4] + v[t1][x + 5][y + 4]))*mu_x[x + 4][y + 4])*(damp[x + 4][y + 4] + damp[x + 5][y + 4]);
        tau_zy[t1][x + 4][y + 4] = 5.0e-1F*dt*(r1*tau_zy[t0][x + 4][y + 4] + (1.04166667e-1F*(v[t1][x + 4][y + 3] - v[t1][x + 4][y + 6]) + 2.81250F*(-v[t1][x + 4][y + 4] + v[t1][x + 4][y + 5]))*mu_z[x + 4][y + 4])*(damp[x + 4][y + 4] + damp[x + 4][y + 5]);
      }
    }
    STOP(section0,timers)

    START(section1)
    if (src_vec->size[0]*src_vec->size[1] > 0 && p_src_M - p_src_m + 1 > 0)
    {
      #pragma acc parallel loop collapse(3) present(src,src_coords,v)
      for (int p_src = p_src_m; p_src <= p_src_M; p_src += 1)
      {
        for (int rsrcx = 0; rsrcx <= 1; rsrcx += 1)
        {
          for (int rsrcy = 0; rsrcy <= 1; rsrcy += 1)
          {
            int posx = static_cast<int>(std::floor(2.50*(-o_x + src_coords[p_src][0])));
            int posy = static_cast<int>(std::floor(2.50*(-o_y + src_coords[p_src][1])));
            float px = 2.50F*(-o_x + src_coords[p_src][0]) - std::floor(2.50F*(-o_x + src_coords[p_src][0]));
            float py = 2.50F*(-o_y + src_coords[p_src][1]) - std::floor(2.50F*(-o_y + src_coords[p_src][1]));
            if (rsrcx + posx >= x_m - 1 && rsrcy + posy >= y_m - 1 && rsrcx + posx <= x_M + 1 && rsrcy + posy <= y_M + 1)
            {
              float r0 = dt*(rsrcx*px + (1 - rsrcx)*(1 - px))*(rsrcy*py + (1 - rsrcy)*(1 - py))*src[time][p_src];
              #pragma acc atomic update
              v[t1][rsrcx + posx + 4][rsrcy + posy + 4] += r0;
            }
          }
        }
      }
    }
    STOP(section1,timers)

    START(section2)
    if (rec_vec->size[0]*rec_vec->size[1] > 0 && p_rec_M - p_rec_m + 1 > 0)
    {
      #pragma acc parallel loop present(rec,rec_coords,v)
      for (int p_rec = p_rec_m; p_rec <= p_rec_M; p_rec += 1)
      {
        float r4 = 2.50F*(-o_x + rec_coords[p_rec][0]);
        float r2 = std::floor(r4);
        int posx = static_cast<int>(r2);
        float r5 = 2.50F*(-o_y + rec_coords[p_rec][1]);
        float r3 = std::floor(r5);
        int posy = static_cast<int>(r3);
        float px = -r2 + r4;
        float py = -r3 + r5;
        float sum = 0.0F;

        for (int rrecx = 0; rrecx <= 1; rrecx += 1)
        {
          for (int rrecy = 0; rrecy <= 1; rrecy += 1)
          {
            if (rrecx + posx >= x_m - 1 && rrecy + posy >= y_m - 1 && rrecx + posx <= x_M + 1 && rrecy + posy <= y_M + 1)
            {
              sum += (rrecx*px + (1 - rrecx)*(1 - px))*(rrecy*py + (1 - rrecy)*(1 - py))*v[t0][rrecx + posx + 4][rrecy + posy + 4];
            }
          }
        }

        rec[time][p_rec] = sum;
      }
    }
    STOP(section2,timers)
  }

  #pragma acc exit data copyout(rec[0:rec_vec->size[0]][0:rec_vec->size[1]])
  #pragma acc exit data delete(rec[0:rec_vec->size[0]][0:rec_vec->size[1]]) if(devicerm)
  #pragma acc exit data copyout(tau_xy[0:tau_xy_vec->size[0]][0:tau_xy_vec->size[1]][0:tau_xy_vec->size[2]])
  #pragma acc exit data delete(tau_xy[0:tau_xy_vec->size[0]][0:tau_xy_vec->size[1]][0:tau_xy_vec->size[2]]) if(devicerm)
  #pragma acc exit data copyout(tau_zy[0:tau_zy_vec->size[0]][0:tau_zy_vec->size[1]][0:tau_zy_vec->size[2]])
  #pragma acc exit data delete(tau_zy[0:tau_zy_vec->size[0]][0:tau_zy_vec->size[1]][0:tau_zy_vec->size[2]]) if(devicerm)
  #pragma acc exit data copyout(v[0:v_vec->size[0]][0:v_vec->size[1]][0:v_vec->size[2]])
  #pragma acc exit data delete(v[0:v_vec->size[0]][0:v_vec->size[1]][0:v_vec->size[2]]) if(devicerm)
  #pragma acc exit data delete(b[0:b_vec->size[0]][0:b_vec->size[1]]) if(devicerm)
  #pragma acc exit data delete(damp[0:damp_vec->size[0]][0:damp_vec->size[1]]) if(devicerm)
  #pragma acc exit data delete(mu_x[0:mu_x_vec->size[0]][0:mu_x_vec->size[1]]) if(devicerm)
  #pragma acc exit data delete(mu_z[0:mu_z_vec->size[0]][0:mu_z_vec->size[1]]) if(devicerm)
  #pragma acc exit data delete(rec_coords[0:rec_coords_vec->size[0]][0:rec_coords_vec->size[1]]) if(devicerm)
  #pragma acc exit data delete(src[0:src_vec->size[0]][0:src_vec->size[1]]) if(devicerm)
  #pragma acc exit data delete(src_coords[0:src_coords_vec->size[0]][0:src_coords_vec->size[1]]) if(devicerm)

  return 0;
}

