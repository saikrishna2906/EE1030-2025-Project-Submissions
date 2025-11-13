  #include <stdio.h>
  #include <stdlib.h>
  #include <math.h>
  #define STB_IMAGE_IMPLEMENTATION
  #include "stb_image.h"
  #define STB_IMAGE_WRITE_IMPLEMENTATION
  #include "stb_image_write.h"

  static double randn(void){
      double u1 = ((double)rand()+1.0)/((double)RAND_MAX+2.0);
      double u2 = ((double)rand()+1.0)/((double)RAND_MAX+2.0);
      return sqrt(-2.0*log(u1))*cos(2.0*M_PI*u2);
  }

  static void compute_BBt(const double *B, int l, int n, double *S){
      for(int i=0;i<l;i++){
          for(int j=0;j<l;j++){
              double s=0.0;
              for(int t=0;t<n;t++) s += B[i*n + t] * B[j*n + t];
              S[i*l + j] = s;}}}

  static void topk_eigs_deflated(double *S_in, int l, int k, int maxit, double tol, double *eig, double *Usmall){
  double *S = (double*)malloc(l*l*sizeof(double));
  for(int i=0;i<l*l;i++) S[i] = S_in[i];
      for(int comp=0; comp<k; ++comp){
          double *x = (double*)malloc(l*sizeof(double));
            for(int i=0;i<l;i++) x[i] = randn();
          double nrm=0; for(int i=0;i<l;i++) nrm += x[i]*x[i]; nrm = sqrt(nrm); if(nrm<1e-12) nrm=1e-12;
              for(int i=0;i<l;i++) x[i] /= nrm;
  double lambda = 0.0;
          for(int it=0; it<maxit; ++it){
              double *y = (double*)malloc(l*sizeof(double));
              for(int i=0;i<l;i++){
                  double s=0.0;
                  for(int j=0;j<l;j++) s += S[i*l + j] * x[j];
                  y[i]=s;
              }
  double n2=0; 
  for(int i=0;i<l;i++) n2 += y[i]*y[i]; n2 = sqrt(n2); if(n2<1e-12) n2=1e-12;
    for(int i=0;i<l;i++) y[i] /= n2;
  double lam_new = 0.0;
      for(int i=0;i<l;i++){
  double s=0;
  for(int j=0;j<l;j++) s += y[j] * (S[j*l + i]); 
         lam_new += y[i] * s;}
  if (fabs(lam_new - lambda) < tol * fabs(lam_new+1e-16)) {
  for(int i=0;i<l;i++) x[i] = y[i];
  lambda = lam_new;
  free(y);
  break;
  }
  lambda = lam_new;
  for(int i=0;i<l;i++){ x[i] = y[i]; }
              free(y);
          }
   eig[comp] = lambda;
          for(int i=0;i<l;i++) Usmall[i*k + comp] = x[i];
  for(int i=0;i<l;i++) for(int j=0;j<l;j++) S[i*l + j] -= lambda * x[i] * x[j];
  free(x);
      }
      free(S);
  }
  static void matmul(const double *A, int m, int p, const double *B, int n, double *C){
      for(int i=0;i<m;i++){
          for(int j=0;j<n;j++){
              double s=0.0;
              for(int t=0;t<p;t++) s += A[i*p + t] * B[t*n + j];
              C[i*n + j] = s;
          }}}

  int main(int argc,char **argv){
  if(argc<3){ fprintf(stderr,"Usage: %s in.png out.png [k] [q]\n", argv[0]); return 1; }
  const char *in = argv[1], *outname = argv[2];
  int k = (argc>3)?atoi(argv[3]):50, q = (argc>4)?atoi(argv[4]):2;
  int w,h,comp;
  unsigned char *img = stbi_load(in,&w,&h,&comp,1);
  if(!img){ fprintf(stderr,"load failed\n"); return 1; }
  int m=h, n=w;
  fprintf(stderr,"Loaded %s (%d x %d)\n", in, w, h);

  int mn = m*n;
  double *A = (double*)calloc(mn, sizeof(double));
  double *A_approx = (double*)calloc(mn, sizeof(double));
  for(int r=0;r<m;r++) for(int c=0;c<n;c++) A[r*n + c] = (double)img[r*w + c];
  stbi_image_free(img);

  if(k<1) k=1; if(k> (m<n?m:n)) k=(m<n?m:n);
  int p = 10;
  int l = k + p; if(l > (m<n?m:n)) l = (m<n?m:n);

  double *Omega = (double*)malloc(n * l * sizeof(double));
  double *Y = (double*)calloc(m * l, sizeof(double));
  double *Z = (double*)calloc(n * l, sizeof(double));
  double *B = (double*)calloc(l * n, sizeof(double));
  if(!Omega||!Y||!Z||!B){ fprintf(stderr,"alloc fail\n"); return 1; }
  srand(42);
  for(int i=0;i<n*l;i++) Omega[i]=randn();
      for(int i=0;i<m;i++){
          for(int j=0;j<l;j++){
              double s=0.0;
              for(int t=0;t<n;t++) s += A[i*n + t] * Omega[t*l + j];
              Y[i*l + j] = s;
          }
      }
  for(int it=0; it<q; ++it){
        for(int i=0;i<n;i++) for(int j=0;j<l;j++){
            double s=0.0;
              for(int r=0;r<m;r++) s += A[r*n + i] * Y[r*l + j];
              Z[i*l + j] = s;
          }
  for(int i=0;i<m;i++) for(int j=0;j<l;j++){
             double s=0.0;
              for(int t=0;t<n;t++) s += A[i*n + t] * Z[t*l + j];
              Y[i*l + j] = s;
          }
      }

   for(int col=0; col<l; ++col){
      for(int prev=0; prev<col; ++prev){
          double dot=0.0;
           for(int r=0;r<m;r++) dot += Y[r*l + prev] * Y[r*l + col];
              for(int r=0;r<m;r++) Y[r*l + col] -= dot * Y[r*l + prev];
          }
   double nrm=0.0;
      for(int r=0;r<m;r++) nrm += Y[r*l + col] * Y[r*l + col];
          nrm = sqrt(nrm); if(nrm<1e-12) nrm=1e-12;
          for(int r=0;r<m;r++) Y[r*l + col] /= nrm;
      }

    for(int i=0;i<l;i++) for(int t=0;t<n;t++){
        double s=0.0;
          for(int r=0;r<m;r++) s += Y[r*l + i] * A[r*n + t];
          B[i*n + t] = s;
      }

  double *S = (double*)malloc(l * l * sizeof(double));
  compute_BBt(B, l, n, S);
  double *eig = (double*)malloc(k * sizeof(double));
  double *Usmall = (double*)calloc(l * k, sizeof(double));
  topk_eigs_deflated(S, l, k, 1000, 1e-8, eig, Usmall);
  double *Sigma = (double*)malloc(k * sizeof(double));
  double *Vt = (double*)calloc(k * n, sizeof(double));
  for(int i=0;i<k;i++){
  double lam = eig[i];
    if(lam < 0) lam = 0;
    double sigma = sqrt(lam);
    if(sigma < 1e-12) sigma = 1e-12;
     Sigma[i] = sigma;
      for(int col=0; col<n; ++col){
         double s = 0.0;
          for(int r=0;r<l;r++) s += Usmall[r*k + i] * B[r*n + col];
              Vt[i*n + col] = s / sigma;
          }
      }
  double *M = (double*)calloc(m * k, sizeof(double));
      for(int r=0;r<m;r++){
          for(int i=0;i<k;i++){
              double s = 0.0;
              for(int t=0;t<l;t++) s += Y[r*l + t] * Usmall[t*k + i];
              M[r*k + i] = s;
          }
      }
  double *temp = (double*)calloc(k * n, sizeof(double));
  for(int i=0;i<k;i++) for(int j=0;j<n;j++) temp[i*n + j] = Sigma[i] * Vt[i*n + j];
      matmul(M, m, k, temp, n, A_approx);
  double acc=0.0;
  for(int i=0;i<mn;i++){ double d = A[i] - A_approx[i]; acc += d*d; }
      fprintf(stderr,"||A - A_k||_F = %.6e\n", sqrt(acc));
      unsigned char *out = (unsigned char*)malloc(mn);
      for(int r=0;r<m;r++) for(int c=0;c<n;c++){
          double v = A_approx[r*n + c];
          if(v<0) v=0; if(v>255) v=255;
          out[r*w + c] = (unsigned char)(v + 0.5);
      }
      stbi_write_png(outname, w, h, 1, out, w);
      fprintf(stderr,"Saved %s\n", outname);
      free(A); free(A_approx); free(Omega); free(Y); free(Z); free(B);
      free(S); free(eig); free(Usmall); free(Sigma); free(Vt); free(M); free(temp); free(out);
      return 0;
  }

