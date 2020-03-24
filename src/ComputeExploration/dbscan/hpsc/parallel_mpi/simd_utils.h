#ifndef SIMD_UTILS_H
#define SIMD_UTILS_H

#include "arm_neon.h"

#include "stdio.h"

#define _DEBUG_SIMD_UTILS_H 0

void simd_squared_dist(const float * const p1, const float * const p2, int dims, float *dist) {
#if _DEBUG_SIMD_UTILS_H
  printf("\n=========== ENTERING SIMD_SQUARED_DIST ===============\n");
#endif
  *dist = 0.0;
  float32x4_t dist_vec = vdupq_n_f32(0.0);
  float32x4_t tmp_vec;
  float32x4_t p1_vec;
  float32x4_t p2_vec;
  
  for(int k = 0; k < dims/4; ++k) {
    //vectorize
    p1_vec = vld1q_f32(&p1[4*k]);
    p2_vec = vld1q_f32(&p2[4*k]);    
    //sub
    tmp_vec = vsubq_f32(p1_vec, p2_vec);
    //square and accumulate
    dist_vec = vmlaq_f32(dist_vec, tmp_vec, tmp_vec);
  }
  
  //reduce vector
  *dist += vgetq_lane_f32(dist_vec, 0);
  if(_DEBUG_SIMD_UTILS_H) {
    printf("Adding lane 1: %f\n", *dist);
  }
  *dist += vgetq_lane_f32(dist_vec, 1);
  if(_DEBUG_SIMD_UTILS_H) {
    printf("Adding lane 2: %f\n", *dist);
  }
  *dist += vgetq_lane_f32(dist_vec, 2);
  if(_DEBUG_SIMD_UTILS_H) {
    printf("Adding lane 3: %f\n", *dist);
  }
  *dist += vgetq_lane_f32(dist_vec, 3);
  if(_DEBUG_SIMD_UTILS_H) {
    printf("Adding lane 4: %f\n", *dist);
  }
  
  //Last 3 dims (if dims % 4 != 0)
  for(int k = dims/4 * 4; k < dims; ++k) {
    *dist += (p1[k] - p2[k]) * (p1[k] - p2[k]);
  }
  
  if(_DEBUG_SIMD_UTILS_H) {
    printf("Adding remaining dims: %f\n", *dist);
  }
#if _DEBUG_SIMD_UTILS_H
  printf("============ EXITING SIMD_SQUARED_DIST ===============\n");
#endif
}

//Only call with check_freq > 0
void simd_squared_dist(const float * const p1, const float * const p2, int dims, float *dist, double eps, int check_freq) {
  *dist = 0.0;
  float32x4_t dist_vec;
  float32x4_t tmp_vec;
  float32x4_t p1_vec;
  float32x4_t p2_vec;

#if _DEBUG_SIMD_UTILS_H
  int dims_counted = 0;
  printf("\n=========== ENTERING SIMD_SQUARED_DIST ===============\n");
#endif
  
  //Sum 4 dimensions at a time with check against epsilon every check_freq iterations
  for(int i = 0; i < dims/4/check_freq; ++i) {
    dist_vec = vdupq_n_f32(0.0);
    for(int k = 0; k < check_freq; ++k) {
      if(_DEBUG_SIMD_UTILS_H) {
	printf("Summed 4 dimensions starting at %d\n", 4*(i * check_freq + k));
      }
      //vectorize
      p1_vec = vld1q_f32(&p1[4*(i * check_freq + k)]);
      p2_vec = vld1q_f32(&p2[4*(i * check_freq + k)]);
      //sub
      tmp_vec = vsubq_f32(p1_vec, p2_vec);
      //square and accumulate
      dist_vec = vmlaq_f32(dist_vec, tmp_vec, tmp_vec);
#if _DEBUG_SIMD_UTILS_H
      dims_counted += 4;
#endif
    }
    
    //reduce vector
    *dist += vgetq_lane_f32(dist_vec, 0);
    if(_DEBUG_SIMD_UTILS_H) {
      printf("Adding lane 1: %f\n", *dist);
    }
    *dist += vgetq_lane_f32(dist_vec, 1);
    if(_DEBUG_SIMD_UTILS_H) {
      printf("Adding lane 2: %f\n", *dist);
    }
    *dist += vgetq_lane_f32(dist_vec, 2);
    if(_DEBUG_SIMD_UTILS_H) {
      printf("Adding lane 3: %f\n", *dist);
    }
    *dist += vgetq_lane_f32(dist_vec, 3);
    if(_DEBUG_SIMD_UTILS_H) {
      printf("Adding lane 4: %f\n", *dist);
    }
    //check for early termination
    if(*dist > eps) {
#if _DEBUG_SIMD_UTILS_H
      printf("Current distance %f exceeds eps %f", *dist, eps);
      printf("\n============ EXITING SIMD_SQUARED_DIST ===============\n"); 
#endif
      return;
    }
  }
  
  //Last sets of 4 that won't reach another check_freq
  dist_vec = vdupq_n_f32(0.0);
  for(int k = dims/4/check_freq * check_freq; k < dims/4; ++k) {
    //vectorize
    p1_vec = vld1q_f32(&p1[4*k]);
    p2_vec = vld1q_f32(&p2[4*k]);
    //sub
    tmp_vec = vsubq_f32(p1_vec, p2_vec);
    //square and accumulate
    dist_vec = vmlaq_f32(dist_vec, tmp_vec, tmp_vec);
#if _DEBUG_SIMD_UTILS_H
    dims_counted += 4;
#endif
  }
  
  //reduce vector
  *dist += vgetq_lane_f32(dist_vec, 0);
  if(_DEBUG_SIMD_UTILS_H) {
    printf("Adding lane 1: %f\n", *dist);
  }
  *dist += vgetq_lane_f32(dist_vec, 1);
  if(_DEBUG_SIMD_UTILS_H) {
    printf("Adding lane 2: %f\n", *dist);
  }
  *dist += vgetq_lane_f32(dist_vec, 2);
  if(_DEBUG_SIMD_UTILS_H) {
    printf("Adding lane 3: %f\n", *dist);
  }
  *dist += vgetq_lane_f32(dist_vec, 3);
  if(_DEBUG_SIMD_UTILS_H) {
    printf("Adding lane 4: %f\n", *dist);
  }
  
  //Last 3 dims (if dims % 4 != 0)
  for(int k = dims/4 * 4; k < dims; ++k) {
    *dist += (p1[k] - p2[k]) * (p1[k] - p2[k]);
#if _DEBUG_SIMD_UTILS_H
      dims_counted += 1;
#endif
  }
  
  if(_DEBUG_SIMD_UTILS_H) {
    printf("Adding remaining dims: %f\n", *dist);
  }
#if _DEBUG_SIMD_UTILS_H
  if(dims_counted != dims) {
    printf("WRONG NUMBER OF DIMESNSIONS SUMMED: %d/%d\n", dims_counted, dims);
  }
  printf("\n============ EXITING SIMD_SQUARED_DIST ===============\n");
#endif
}


#endif
