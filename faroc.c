/***********************************************************/
/*Source Code of FaRoC Algorithm written in C and R        */
/*A. Mandal and P. Maji, FaRoC: Fast and Robust Supervised */
/*Canonical Correlation Analysis for Multimodal Omics Data,*/
/*IEEE Transactions on Cybernetics, 2018.                  */
/***********************************************************/

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<assert.h>
#include<time.h>
#include<sys/timeb.h>
#include<sys/types.h>
#include<sys/stat.h>
#include<unistd.h>
#include<errno.h>

struct dataset
{
	double **data_matrix;
	int *class_labels;
	int number_of_samples;
	int number_of_features;
	int number_of_class_labels;
};

struct basis_vector
{
	double **basis_vector_data_matrix1;
	double **basis_vector_data_matrix2;
};

struct featureset
{
	double **basis_vector_data_matrix1;
	double **basis_vector_data_matrix2;
	double **canonical_variables_data_matrix;
	double *correlation;
	struct feature *rhepm_feature;
};

struct feature
{
	int **rhepm_data_matrix;
	double relevance;
	double significance;
	double objective_function_value;
};

void write_instruction(void);
double **double_matrix_allocation(int,int);
int **int_matrix_allocation(int,int);
void preprocessing(char *,char *,char *,int,double,double,double,double,double);
struct dataset read_input_file(char *);
void read_eigenvector_file(char *,double **,int,int);
void read_eigenvalue_file(char *,double *,int);
void zero_mean(double **,double **,int,int);
void FaRoC(char *,double **,double **,int *,double,double,double,int,int,int,int,int,double,double);
void between_set_covariance(double **,double **,double **,int,int,int);
void within_set_covariance(double **,double **,int,int);
void matrix_transpose(double **,double **,int,int);
void matrix_multiplication(double **,double **,double **,int,int,int);
void eigenvalue_eigenvector(char *,double **,double **,double *,int);
void eigenvalue_eigenvector_power_method(char *,double **,double **,double *,int,int,int,int);
double rhepm(struct featureset,int **,double *,int *,int,int,int,double,double);
void generate_equivalence_partition(int **,double *,int *,int,int,double);
double dependency_degree(int **,int,int);
double form_resultant_equivalence_partition_matrix(int **,int **,int **,int,int);
void write_output_file(char *,double **,int *,int,int,int);
void write_file(char *,double **,int,int);
void write_correlation_file(char *,double *,int);
void write_R_eigenvalue_eigenvector(char *,char *,char *,char *);
void double_matrix_deallocation(double **,int);
void int_matrix_deallocation(int **,int);

int main(int argc, char *argv[])
{
	int o;
	extern char *optarg;
	char *filename1=NULL,*filename2=NULL,*path;
	int number_of_new_features;
	double lambda_minimum,lambda_maximum,delta;
	double epsilon,omega;
	time_t t;
	struct timeb ti,tf;
	size_t allocsize=sizeof(char)*1024;

	number_of_new_features=10;
	lambda_minimum=0.0;
	lambda_maximum=1.0;
	delta=0.1;
	epsilon=0.0;
	omega=0.5;

	path=(char *)malloc(allocsize);
	if(getcwd(path,allocsize)!=NULL)
       		fprintf(stdout,"Current Working Directory: %s\n",path);
   	else
	{
       		perror("getcwd() error");
		exit(0);
	}
	strcat(path,"/");

	while((o=getopt(argc,argv,"1:2:f:p:m:n:d:o:h"))!=EOF)
	{
		switch(o)
		{
			case '1': filename1=optarg;
				  printf("\tFile stem1 <%s>\n",filename1);
				  break;
			case '2': filename2=optarg;
				  printf("\tFile stem2 <%s>\n",filename2);
				  break;
                      	case 'f': number_of_new_features=atoi(optarg);
                                  printf("\tNumber of New Features: %d\n",number_of_new_features);
                                  break;
                        case 'p': path=optarg;
				  printf("\tPath for Input/Output Files: <%s>\n",path);
				  break;
                        case 'm': lambda_minimum=atof(optarg);
                                  printf("\tMinimum Value of Regularization Parameter: %f\n",lambda_minimum);
                                  break;
                        case 'n': lambda_maximum=atof(optarg);
                                  printf("\tMaximum Value of Regularization Parameter: %f\n",lambda_maximum);
                                  break;
                        case 'd': delta=atof(optarg);
                                  printf("\tIncrement of Regularization Parameter: %f\n",delta);
                                  break;
			case 'o': omega=atof(optarg);
				  printf("\tWeight Parameter: %lf\n",omega);
				  break;
			case 'h': write_instruction();
				  break;
			case '?': printf("Unrecognised option\n");
				  exit(1);
		}
	}
	if(filename1==NULL||filename2==NULL||omega<0||omega>1)
		write_instruction();
	(void)ftime(&ti);
	preprocessing(filename1,filename2,path,number_of_new_features,lambda_minimum,lambda_maximum,delta,epsilon,omega);
	(void)ftime(&tf);
	printf("\nTOTAL TIME REQUIRED for FaRoC=%d millisec\n",(int)(1000.0*(tf.time-ti.time)+(tf.millitm-ti.millitm)));
	free(path);
	printf("\n");
}

void write_instruction(void)
{
	system("clear");
	printf("1:\tInput File 1\n");
	printf("2:\tInput File 2\n");
	printf("f:\tNumber of New Features\n");
	printf("p:\tPath for Input/Output Files\n");
	printf("m:\tMinimum Value of Regularization Parameter\n");
	printf("n:\tMaximum Value of Regularization Parameter\n");
	printf("d:\tIncrement of Regularization Parameter\n");
	printf("o:\tWeight Parameter\n");
	printf("h:\tHelp\n");
	exit(1);
}

int **int_matrix_allocation(int row,int column)
{
        int i;
        int **data;

        data=(int **)malloc(sizeof(int *)*row);
        assert(data!=NULL);
        for(i=0;i<row;i++)
        {
                data[i]=(int *)malloc(sizeof(int)*column);
                assert(data[i]!=NULL);
        }
        return data;
}

double **double_matrix_allocation(int row,int column)
{
        int i;
        double **data;

        data=(double **)malloc(sizeof(double *)*row);
        assert(data!=NULL);
        for(i=0;i<row;i++)
        {
                data[i]=(double *)malloc(sizeof(double)*column);
                assert(data[i]!=NULL);
        }
        return data;
}

void preprocessing(char *filename1,char *filename2,char *path,int number_of_new_features,double lambda_minimum,double lambda_maximum,double delta,double epsilon,double omega)
{
	struct dataset dataset1,dataset2,temp_dataset;
	double **transpose_data_matrix1,**transpose_data_matrix2;
	double **zero_mean_data_matrix1,**zero_mean_data_matrix2;
	char *filename;
	int i,j;
	struct stat st={0};
			
	filename=(char *)malloc(sizeof(char)*1000);		
	strcpy(filename,path);
	strcat(filename,filename1);
	dataset1=read_input_file(filename);
	free(filename);

	filename=(char *)malloc(sizeof(char)*1000);		
	strcpy(filename,path);
	strcat(filename,filename2);
	dataset2=read_input_file(filename);
	free(filename);

	if(dataset1.number_of_samples!=dataset2.number_of_samples)
	{
		printf("\nError: Number of Samples in Dataset1 = %d\tNumber of Samples in Dataset2 = %d", dataset1.number_of_samples,dataset2.number_of_samples);
		exit(0);
	}
	for(j=0;j<dataset1.number_of_samples;j++)
	{
		if(dataset1.class_labels[j]!=dataset2.class_labels[j])
		{
			printf("\nError: Class labels1[%d] = %d\tClass labels2[%d] = %d",j+1,dataset1.class_labels[j],j+1,dataset2.class_labels[j]);
			exit(0);
		}
	}
	if(dataset1.number_of_features>dataset2.number_of_features)
	{
		if(number_of_new_features>dataset2.number_of_features)
		{
			printf("\nError: Number of Extracted Features = %d\tMinimum Number of Input Features = %d",number_of_new_features,dataset2.number_of_features);
			exit(0);
		}
		temp_dataset=dataset1;
		dataset1=dataset2;
		dataset2=temp_dataset; 
	}
	else
	{
		if(number_of_new_features>dataset1.number_of_features)
		{
			printf("\nError: Number of Extracted Features = %d\tMinimum Number of Input Features = %d",number_of_new_features,dataset1.number_of_features);
			exit(0);
		}
	}
	transpose_data_matrix1=double_matrix_allocation(dataset1.number_of_features,dataset1.number_of_samples);
	matrix_transpose(dataset1.data_matrix,transpose_data_matrix1,dataset1.number_of_features,dataset1.number_of_samples);

	transpose_data_matrix2=double_matrix_allocation(dataset2.number_of_features,dataset2.number_of_samples);
	matrix_transpose(dataset2.data_matrix,transpose_data_matrix2,dataset2.number_of_features,dataset2.number_of_samples);

	zero_mean_data_matrix1=double_matrix_allocation(dataset1.number_of_features,dataset1.number_of_samples);
	zero_mean(transpose_data_matrix1,zero_mean_data_matrix1,dataset1.number_of_features,dataset1.number_of_samples);

	zero_mean_data_matrix2=double_matrix_allocation(dataset2.number_of_features,dataset2.number_of_samples);
	zero_mean(transpose_data_matrix2,zero_mean_data_matrix2,dataset2.number_of_features,dataset2.number_of_samples);

	filename=(char *)malloc(sizeof(char)*1000);
	strcpy(filename,path);
	strcat(filename,"FaRoC/");
	if(stat(filename,&st)==-1)
	    	mkdir(filename,0700);

	FaRoC(filename,zero_mean_data_matrix1,zero_mean_data_matrix2,dataset1.class_labels,lambda_minimum,lambda_maximum,delta,dataset1.number_of_samples,dataset1.number_of_features,dataset2.number_of_features,number_of_new_features,dataset1.number_of_class_labels,epsilon,omega);

	double_matrix_deallocation(dataset1.data_matrix,dataset1.number_of_samples);
	double_matrix_deallocation(dataset2.data_matrix,dataset2.number_of_samples);
	double_matrix_deallocation(transpose_data_matrix1,dataset1.number_of_features);
	double_matrix_deallocation(transpose_data_matrix2,dataset2.number_of_features);
	double_matrix_deallocation(zero_mean_data_matrix1,dataset1.number_of_features);
	double_matrix_deallocation(zero_mean_data_matrix2,dataset2.number_of_features);
	free(dataset1.class_labels);
	free(dataset2.class_labels);
	free(filename);
}

struct dataset read_input_file(char *filename)
{
	struct dataset Dataset;
	int i,j; 
	FILE *fp_read;  
	
	fp_read=fopen(filename,"r");
 	if(filename==NULL)
  	{
		printf("\nError: Error in Input File.\n");
		exit(0);
  	}
  	fscanf(fp_read,"%d%d%d",&Dataset.number_of_samples,&Dataset.number_of_features,&Dataset.number_of_class_labels);    
	Dataset.data_matrix=double_matrix_allocation(Dataset.number_of_samples,Dataset.number_of_features);
	Dataset.class_labels=(int *)malloc(sizeof(int)*Dataset.number_of_samples);
  	for(i=0;i<Dataset.number_of_samples;i++)
  	{
  		for(j=0;j<Dataset.number_of_features;j++)
			fscanf(fp_read,"%lf",&Dataset.data_matrix[i][j]);
		fscanf(fp_read,"%d",&Dataset.class_labels[i]);
	}
  	fclose(fp_read);
  	return Dataset;
}

void read_eigenvector_file(char *filename,double **data_matrix,int row,int column)
{
	FILE *fp_read; 
	int i,j; 
	
	fp_read=fopen(filename,"r");
 	if(filename==NULL)
  	{
		printf("\nError: Error in Input File.\n");
		exit(0);
  	}  
  	for(i=0;i<row;i++)
  		for(j=0;j<column;j++)
			fscanf(fp_read,"%lf",&data_matrix[i][j]);
  	fclose(fp_read);
}

void read_eigenvalue_file(char *filename,double *data_matrix,int size_of_matrix)
{
	FILE *fp_read; 
	int i; 
	
	fp_read=fopen(filename,"r");
 	if(filename==NULL)
  	{
		printf("\nError: Error in Input File.\n");
		exit(0);
  	}  
  	for(i=0;i<size_of_matrix;i++)
		fscanf(fp_read,"%lf",&data_matrix[i]);
  	fclose(fp_read);
}

void zero_mean(double **data_matrix,double **new_data_matrix,int row,int column)
{
	double *mean;
	double sum;
	int i,j;
		
	mean=(double *)malloc(sizeof(double)*row);

	for(i=0;i<row;i++)
	{
		sum=0;
		for(j=0;j<column;j++)
			sum+=data_matrix[i][j];
		mean[i]=sum/column;
	}
	for(i=0;i<row;i++)
		for(j=0;j<column;j++)
			new_data_matrix[i][j]=data_matrix[i][j]-mean[i];

	free(mean);
}

void FaRoC(char *path,double **data_matrix1,double **data_matrix2,int *class_labels,double lambda_minimum,double lambda_maximum,double delta,int number_of_samples,int number_of_features1,int number_of_features2,int number_of_new_features,int number_of_class_labels,double epsilon,double omega)
{
	struct basis_vector *basis_vector_set;
	struct featureset optimal_featureset;
	struct feature *new_feature;
	double ***h_data_matrix1,***h_data_matrix2;
	double ***new_h_data_matrix;
	double **cross_covariance_data_matrix1,**cross_covariance_data_matrix2;
	double **covariance_data_matrix1,**covariance_data_matrix2;
	double **inverse_covariance_data_matrix1,**inverse_covariance_data_matrix2;
	double **eigenvector_of_covariance_data_matrix1,**eigenvector_of_covariance_data_matrix2;
	double **transpose_eigenvector_data_matrix1,**transpose_eigenvector_data_matrix2;
	double **temp_data_matrix,**multiplication_data_matrix;
	double **eigenvalue_data_matrix;
	double **canonical_variables_data_matrix;
	double *eigenvalue_of_covariance_data_matrix1,*eigenvalue_of_covariance_data_matrix2;
	double *cca_variables_data_matrix1,*cca_variables_data_matrix2;
	double *correlation;
	double *objective_function_value;
	double maximum_objective_function_value;
	double value;
	char *filename;
	int count;
	int index;
	int i,j,k,l,t,m,n;

	count=(int)((lambda_maximum-lambda_minimum)/delta)+1;
	if(!number_of_new_features)
		number_of_new_features=number_of_features1;

	cross_covariance_data_matrix1=double_matrix_allocation(number_of_features1,number_of_features2);
	between_set_covariance(data_matrix1,data_matrix2,cross_covariance_data_matrix1,number_of_features1,number_of_features2,number_of_samples);

	cross_covariance_data_matrix2=double_matrix_allocation(number_of_features2,number_of_features1);
	matrix_transpose(cross_covariance_data_matrix1,cross_covariance_data_matrix2,number_of_features2,number_of_features1);

	covariance_data_matrix1=double_matrix_allocation(number_of_features1,number_of_features1);		
	within_set_covariance(data_matrix1,covariance_data_matrix1,number_of_features1,number_of_samples);
	if(lambda_minimum)
	{
		for(i=0;i<number_of_features1;i++)
			covariance_data_matrix1[i][i]+=lambda_minimum;
	}

	covariance_data_matrix2=double_matrix_allocation(number_of_features2,number_of_features2);
	within_set_covariance(data_matrix2,covariance_data_matrix2,number_of_features2,number_of_samples);
	if(lambda_minimum)
	{
		for(i=0;i<number_of_features2;i++)
			covariance_data_matrix2[i][i]+=lambda_minimum;
	}

	eigenvalue_of_covariance_data_matrix1=(double *)malloc(sizeof(double)*number_of_features1);
	eigenvector_of_covariance_data_matrix1=double_matrix_allocation(number_of_features1,number_of_features1);
	eigenvalue_eigenvector(path,covariance_data_matrix1,eigenvector_of_covariance_data_matrix1,eigenvalue_of_covariance_data_matrix1,number_of_features1);
	transpose_eigenvector_data_matrix1=double_matrix_allocation(number_of_features1,number_of_features1);
	matrix_transpose(eigenvector_of_covariance_data_matrix1,transpose_eigenvector_data_matrix1,number_of_features1,number_of_features1);
	double_matrix_deallocation(covariance_data_matrix1,number_of_features1);

	eigenvalue_of_covariance_data_matrix2=(double *)malloc(sizeof(double)*number_of_features2);
	eigenvector_of_covariance_data_matrix2=double_matrix_allocation(number_of_features2,number_of_features2);
	eigenvalue_eigenvector(path,covariance_data_matrix2,eigenvector_of_covariance_data_matrix2,eigenvalue_of_covariance_data_matrix2,number_of_features2);
	transpose_eigenvector_data_matrix2=double_matrix_allocation(number_of_features2,number_of_features2);
	matrix_transpose(eigenvector_of_covariance_data_matrix2,transpose_eigenvector_data_matrix2,number_of_features2,number_of_features2);
	double_matrix_deallocation(covariance_data_matrix2,number_of_features2);

	h_data_matrix1=(double ***)malloc(sizeof(double **)*count);
	h_data_matrix2=(double ***)malloc(sizeof(double **)*count);
	multiplication_data_matrix=double_matrix_allocation(number_of_features2,number_of_features1);
	for(t=0;t<count;t++)
	{
		temp_data_matrix=double_matrix_allocation(number_of_features1,number_of_features1);
		for(i=0;i<number_of_features1;i++)
			for(j=0;j<number_of_features1;j++)
			{
				if((eigenvalue_of_covariance_data_matrix1[j]+t*delta)>=0.000001)
					temp_data_matrix[i][j]=eigenvector_of_covariance_data_matrix1[i][j]/(eigenvalue_of_covariance_data_matrix1[j]+t*delta);
				else
					temp_data_matrix[i][j]=eigenvector_of_covariance_data_matrix1[i][j]/0.000001;
			}
		inverse_covariance_data_matrix1=double_matrix_allocation(number_of_features1,number_of_features1);
		matrix_multiplication(temp_data_matrix,transpose_eigenvector_data_matrix1,inverse_covariance_data_matrix1,number_of_features1,number_of_features1,number_of_features1);
		h_data_matrix1[t]=double_matrix_allocation(number_of_features1,number_of_features2);
		matrix_multiplication(inverse_covariance_data_matrix1,cross_covariance_data_matrix1,h_data_matrix1[t],number_of_features1,number_of_features1,number_of_features2);
		double_matrix_deallocation(temp_data_matrix,number_of_features1);
		double_matrix_deallocation(inverse_covariance_data_matrix1,number_of_features1);

		temp_data_matrix=double_matrix_allocation(number_of_features2,number_of_features2);
		for(i=0;i<number_of_features2;i++)
			for(j=0;j<number_of_features2;j++)
			{
				if((eigenvalue_of_covariance_data_matrix2[j]+t*delta)>=0.000001)
					temp_data_matrix[i][j]=eigenvector_of_covariance_data_matrix2[i][j]/(eigenvalue_of_covariance_data_matrix2[j]+t*delta);
				else
					temp_data_matrix[i][j]=eigenvector_of_covariance_data_matrix2[i][j]/0.000001;
			}
		inverse_covariance_data_matrix2=double_matrix_allocation(number_of_features2,number_of_features2);
		matrix_multiplication(temp_data_matrix,transpose_eigenvector_data_matrix2,inverse_covariance_data_matrix2,number_of_features2,number_of_features2,number_of_features2);
		h_data_matrix2[t]=double_matrix_allocation(number_of_features2,number_of_features1);
		matrix_multiplication(inverse_covariance_data_matrix2,cross_covariance_data_matrix2,h_data_matrix2[t],number_of_features2,number_of_features2,number_of_features1);
		double_matrix_deallocation(temp_data_matrix,number_of_features2);
		double_matrix_deallocation(inverse_covariance_data_matrix2,number_of_features2);

		if(!t)
			for(i=0;i<number_of_features2;i++)
				for(j=0;j<number_of_features1;j++)
					multiplication_data_matrix[i][j]=h_data_matrix2[t][i][j];
	}
	double_matrix_deallocation(cross_covariance_data_matrix1,number_of_features1);
	double_matrix_deallocation(cross_covariance_data_matrix2,number_of_features2);
	double_matrix_deallocation(eigenvector_of_covariance_data_matrix1,number_of_features1);
	double_matrix_deallocation(eigenvector_of_covariance_data_matrix2,number_of_features2);
	double_matrix_deallocation(transpose_eigenvector_data_matrix1,number_of_features1);
	double_matrix_deallocation(transpose_eigenvector_data_matrix2,number_of_features2);
	free(eigenvalue_of_covariance_data_matrix1);
	free(eigenvalue_of_covariance_data_matrix2);

	new_h_data_matrix=(double ***)malloc(sizeof(double **)*count*count);
	t=0;
	for(i=0;i<count;i++)
	{
		for(j=0;j<count;j++)
		{
			new_h_data_matrix[t]=double_matrix_allocation(number_of_features1,number_of_features1);
			matrix_multiplication(h_data_matrix1[i],h_data_matrix2[j],new_h_data_matrix[t],number_of_features1,number_of_features2,number_of_features1);
			t++;
		}
	}
	for(t=0;t<count;t++)
	{
		double_matrix_deallocation(h_data_matrix1[t],number_of_features1);
		double_matrix_deallocation(h_data_matrix2[t],number_of_features2);
	}
	free(h_data_matrix1);
	free(h_data_matrix2);

	basis_vector_set=(struct basis_vector *)malloc(sizeof(struct basis_vector)*count*count);
	new_feature=(struct feature *)malloc(sizeof(struct feature)*count*count);
	for(t=0;t<count*count;t++)
	{
		basis_vector_set[t].basis_vector_data_matrix1=double_matrix_allocation(number_of_new_features,number_of_features1);
		basis_vector_set[t].basis_vector_data_matrix2=double_matrix_allocation(number_of_new_features,number_of_features2);
		new_feature[t].rhepm_data_matrix=int_matrix_allocation(number_of_class_labels,number_of_samples);
	}

	optimal_featureset.basis_vector_data_matrix1=double_matrix_allocation(number_of_features1,number_of_new_features);
	optimal_featureset.basis_vector_data_matrix2=double_matrix_allocation(number_of_features2,number_of_new_features);
	optimal_featureset.canonical_variables_data_matrix=double_matrix_allocation(number_of_samples,number_of_new_features);
	optimal_featureset.correlation=(double *)malloc(sizeof(double)*number_of_new_features);
	optimal_featureset.rhepm_feature=(struct feature *)malloc(sizeof(struct feature)*number_of_new_features);
	for(i=0;i<number_of_new_features;i++)
		optimal_featureset.rhepm_feature[i].rhepm_data_matrix=int_matrix_allocation(number_of_class_labels,number_of_samples);	
	
	eigenvalue_data_matrix=double_matrix_allocation(count*count,number_of_new_features);
	canonical_variables_data_matrix=double_matrix_allocation(count*count,number_of_samples);
	cca_variables_data_matrix1=(double *)malloc(sizeof(double)*number_of_samples);
	cca_variables_data_matrix2=(double *)malloc(sizeof(double)*number_of_samples);
	correlation=(double *)malloc(sizeof(double)*count*count);
	objective_function_value=(double *)malloc(sizeof(double)*count*count);

	for(t=0;t<number_of_new_features;t++)
	{
		n=0;
		for(i=0;i<count;i++)
		{
			for(j=0;j<count;j++)
			{
				if(!t)
					eigenvalue_eigenvector_power_method(path,new_h_data_matrix[n],basis_vector_set[n].basis_vector_data_matrix1,eigenvalue_data_matrix[n],number_of_features1,number_of_new_features,i+1,j+1);
				correlation[n]=sqrt(eigenvalue_data_matrix[n][t]);
				for(k=0;k<number_of_features2;k++)
				{
					basis_vector_set[n].basis_vector_data_matrix2[t][k]=0;
					for(l=0;l<number_of_features1;l++)
						basis_vector_set[n].basis_vector_data_matrix2[t][k]+=multiplication_data_matrix[k][l]*basis_vector_set[n].basis_vector_data_matrix1[t][l];
				}
				for(k=0;k<number_of_samples;k++)
				{
					cca_variables_data_matrix1[k]=0;
					for(l=0;l<number_of_features1;l++)
						cca_variables_data_matrix1[k]+=basis_vector_set[n].basis_vector_data_matrix1[t][l]*data_matrix1[l][k];
				}
				for(k=0;k<number_of_samples;k++)
				{
					cca_variables_data_matrix2[k]=0;
					for(l=0;l<number_of_features2;l++)
						cca_variables_data_matrix2[k]+=basis_vector_set[n].basis_vector_data_matrix2[t][l]*data_matrix2[l][k];
				}
				for(k=0;k<number_of_samples;k++)
					canonical_variables_data_matrix[n][k]=cca_variables_data_matrix1[k]+cca_variables_data_matrix2[k];
				objective_function_value[n]=rhepm(optimal_featureset,new_feature[n].rhepm_data_matrix,canonical_variables_data_matrix[n],class_labels,number_of_samples,t,number_of_class_labels,epsilon,omega);
				n++;
				printf("\nNumber of Iteration = %d",n);
			}
		}
		maximum_objective_function_value=objective_function_value[0];
		index=0;
		for(i=1;i<count*count;i++)
		{
			if(maximum_objective_function_value<objective_function_value[i])
			{
				maximum_objective_function_value=objective_function_value[i];
				index=i;
			}
		}
		if(!maximum_objective_function_value)
		{
			number_of_new_features=t;
			break;
		}
		for(i=0;i<number_of_features1;i++)
			optimal_featureset.basis_vector_data_matrix1[i][t]=basis_vector_set[index].basis_vector_data_matrix1[t][i];
		for(i=0;i<number_of_features2;i++)
			optimal_featureset.basis_vector_data_matrix2[i][t]=basis_vector_set[index].basis_vector_data_matrix2[t][i];
		for(i=0;i<number_of_samples;i++)
			optimal_featureset.canonical_variables_data_matrix[i][t]=canonical_variables_data_matrix[index][i];
		optimal_featureset.correlation[t]=correlation[index];
		for(i=0;i<number_of_class_labels;i++)
			for(j=0;j<number_of_samples;j++)
				optimal_featureset.rhepm_feature[t].rhepm_data_matrix[i][j]=new_feature[index].rhepm_data_matrix[i][j];
		if(!t)
		{
			for(i=0;i<count*count;i++)
				double_matrix_deallocation(new_h_data_matrix[i],number_of_features1);
			free(new_h_data_matrix);
		}
		printf("\nNumber of feature = %d",t+1);
	}

	for(i=0;i<count*count;i++)
	{
		double_matrix_deallocation(basis_vector_set[i].basis_vector_data_matrix1,number_of_new_features);
		double_matrix_deallocation(basis_vector_set[i].basis_vector_data_matrix2,number_of_new_features);
		int_matrix_deallocation(new_feature[i].rhepm_data_matrix,number_of_class_labels);
	}
	free(basis_vector_set);
	free(new_feature);
	double_matrix_deallocation(multiplication_data_matrix,number_of_features2);
	double_matrix_deallocation(eigenvalue_data_matrix,count*count);
	double_matrix_deallocation(canonical_variables_data_matrix,count*count);
	free(cca_variables_data_matrix1);
	free(cca_variables_data_matrix2);
	free(correlation);
	free(objective_function_value);

	filename=(char *)malloc(sizeof(char)*1000);
	strcpy(filename,path);
	strcat(filename,"basis_vector1.txt");
	write_file(filename,optimal_featureset.basis_vector_data_matrix1,number_of_features1,number_of_new_features);
	free(filename);

	filename=(char *)malloc(sizeof(char)*1000);
	strcpy(filename,path);
	strcat(filename,"basis_vector2.txt");
	write_file(filename,optimal_featureset.basis_vector_data_matrix2,number_of_features2,number_of_new_features);
	free(filename);

	filename=(char *)malloc(sizeof(char)*1000);
	strcpy(filename,path);
	strcat(filename,"canonical_variables.txt");
	write_output_file(filename,optimal_featureset.canonical_variables_data_matrix,class_labels,number_of_samples,number_of_new_features,number_of_class_labels);
	free(filename);

	filename=(char *)malloc(sizeof(char)*1000);
	strcpy(filename,path);
	strcat(filename,"correlation.txt");
	write_correlation_file(filename,optimal_featureset.correlation,number_of_new_features);
	free(filename);

	double_matrix_deallocation(optimal_featureset.basis_vector_data_matrix1,number_of_features1);
	double_matrix_deallocation(optimal_featureset.basis_vector_data_matrix2,number_of_features2);
	double_matrix_deallocation(optimal_featureset.canonical_variables_data_matrix,number_of_samples);
	free(optimal_featureset.correlation);
	for(i=0;i<number_of_new_features;i++)
		int_matrix_deallocation(optimal_featureset.rhepm_feature[i].rhepm_data_matrix,number_of_class_labels);
	free(optimal_featureset.rhepm_feature);
}

void between_set_covariance(double **data_matrix1,double **data_matrix2,double **new_data_matrix,int row1,int row2,int column)
{
	double **transpose_data_matrix;
	int i,j;
	
	transpose_data_matrix=double_matrix_allocation(column,row2);

	matrix_transpose(data_matrix2,transpose_data_matrix,column,row2);
	matrix_multiplication(data_matrix1,transpose_data_matrix,new_data_matrix,row1,column,row2);
	for(i=0;i<row1;i++)
		for(j=0;j<row2;j++)
			new_data_matrix[i][j]=new_data_matrix[i][j]/column;

	double_matrix_deallocation(transpose_data_matrix,column);
}

void within_set_covariance(double **data_matrix,double **new_data_matrix,int row,int column)
{
	double **transpose_data_matrix;
	int i,j;
	
	transpose_data_matrix=double_matrix_allocation(column,row);

	matrix_transpose(data_matrix,transpose_data_matrix,column,row);
	matrix_multiplication(data_matrix,transpose_data_matrix,new_data_matrix,row,column,row);
	for(i=0;i<row;i++)
		for(j=0;j<row;j++)
			new_data_matrix[i][j]=new_data_matrix[i][j]/column;

	double_matrix_deallocation(transpose_data_matrix, column);
}

void matrix_transpose(double **data_matrix,double **new_data_matrix,int row,int column)
{
	int i,j;
	
	for(i=0;i<row;i++)
		for(j=0;j<column;j++)
			new_data_matrix[i][j]=data_matrix[j][i];
}

void matrix_multiplication(double **data_matrix1,double **data_matrix2,double **new_data_matrix,int row1,int column1,int column2)
{
	int i,j,k;
	
	for(i=0;i<row1;i++)
	{
		for(j=0;j<column2;j++)
		{
			new_data_matrix[i][j]=0;
			for(k=0;k<column1;k++)
				new_data_matrix[i][j]+=data_matrix1[i][k]*data_matrix2[k][j];
		}
	}
}

void eigenvalue_eigenvector(char *path,double **data_matrix,double **eigenvector_data_matrix,double *eigenvalue_data_matrix,int size_of_matrix)
{
	char *data_filename,*foldername;
	char *eigenvector_filename,*eigenvalue_filename;

	data_filename=(char *)malloc(sizeof(char)*1000);
	strcpy(data_filename,path);
	strcat(data_filename,"data.txt");
	write_file(data_filename,data_matrix,size_of_matrix,size_of_matrix);

	eigenvector_filename=(char *)malloc(sizeof(char)*1000);
	strcpy(eigenvector_filename,path);
	strcat(eigenvector_filename,"eigenvector.txt");

	eigenvalue_filename=(char *)malloc(sizeof(char)*1000);
	strcpy(eigenvalue_filename,path);
	strcat(eigenvalue_filename,"eigenvalue.txt");

	write_R_eigenvalue_eigenvector(data_filename,eigenvector_filename,eigenvalue_filename,"eigenvalue_eigenvector.R");
	system("R CMD BATCH eigenvalue_eigenvector.R");
	system("cat eigenvalue_eigenvector.Rout");
	read_eigenvector_file(eigenvector_filename,eigenvector_data_matrix,size_of_matrix,size_of_matrix);
	read_eigenvalue_file(eigenvalue_filename,eigenvalue_data_matrix,size_of_matrix);

	foldername=(char *)malloc(sizeof(char)*1000);
	strcpy(foldername,"rm -f ");
	strcat(foldername,data_filename);
	strcat(foldername," ");
	strcat(foldername,eigenvector_filename);
	strcat(foldername," ");
	strcat(foldername,eigenvalue_filename);
	system(foldername);

	free(data_filename);
	free(eigenvector_filename);
	free(eigenvalue_filename);
	free(foldername);	
}

void eigenvalue_eigenvector_power_method(char *path,double **data_matrix,double **eigenvector_data_matrix,double *eigenvalue_data_matrix,int size_of_matrix,int new_size_of_matrix,int i_value,int j_value)
{
	double **eigenvector;
	char *data_filename,*foldername;
	char *eigenvector_filename,*eigenvalue_filename;

	data_filename=(char *)malloc(sizeof(char)*1000);
	strcpy(data_filename,path);
	strcat(data_filename,"data.txt");
	write_file(data_filename,data_matrix,size_of_matrix,size_of_matrix);

	eigenvector_filename=(char *)malloc(sizeof(char)*1000);
	strcpy(eigenvector_filename,path);
	strcat(eigenvector_filename,"eigenvector.txt");

	eigenvalue_filename=(char *)malloc(sizeof(char)*1000);
	strcpy(eigenvalue_filename,path);
	strcat(eigenvalue_filename,"eigenvalue.txt");

	write_R_eigenvalue_eigenvector(data_filename,eigenvector_filename,eigenvalue_filename,"eigenvalue_eigenvector.R");
	system("R CMD BATCH eigenvalue_eigenvector.R");
	system("cat eigenvalue_eigenvector.Rout");
	eigenvector=double_matrix_allocation(size_of_matrix,size_of_matrix);
	read_eigenvector_file(eigenvector_filename,eigenvector,size_of_matrix,size_of_matrix);
	read_eigenvalue_file(eigenvalue_filename,eigenvalue_data_matrix,new_size_of_matrix);

	matrix_transpose(eigenvector,eigenvector_data_matrix,new_size_of_matrix,size_of_matrix);
	double_matrix_deallocation(eigenvector,size_of_matrix);

	foldername=(char *)malloc(sizeof(char)*1000);
	strcpy(foldername,"rm -f ");
	strcat(foldername,data_filename);
	strcat(foldername," ");
	strcat(foldername,eigenvector_filename);
	strcat(foldername," ");
	strcat(foldername,eigenvalue_filename);
	system(foldername);

	free(data_filename);
	free(eigenvector_filename);
	free(eigenvalue_filename);
	free(foldername);	
}

double rhepm(struct featureset optimal_featureset,int **rhepm_data_matrix,double *canonical_variables_data_matrix,int *class_labels,int number_of_samples,int number_of_new_features,int number_of_class_labels,double epsilon,double omega)
{
	int **resultant_equivalence_partition;
	double relevance;
	double significance;
	double joint_dependency;
	double objective_function_value;
	int i,j;

	generate_equivalence_partition(rhepm_data_matrix,canonical_variables_data_matrix,class_labels,number_of_samples,number_of_class_labels,epsilon);
	relevance=dependency_degree(rhepm_data_matrix,number_of_samples,number_of_class_labels);
	resultant_equivalence_partition=int_matrix_allocation(number_of_class_labels,number_of_samples);
	if(!number_of_new_features)
		objective_function_value=relevance;
	else
	{
		j=0;
		significance=0;
		for(i=0;i<number_of_new_features;i++)
		{
			form_resultant_equivalence_partition_matrix(optimal_featureset.rhepm_feature[i].rhepm_data_matrix,rhepm_data_matrix,resultant_equivalence_partition,number_of_class_labels,number_of_samples);
			joint_dependency=dependency_degree(resultant_equivalence_partition,number_of_samples,number_of_class_labels);
			if(joint_dependency-relevance!=0.0)
				j++;
			significance+=joint_dependency-relevance;
		}
		if(j)
		{
			significance/=j;
			objective_function_value=omega*relevance+(1-omega)*significance;
		}
		else
		{
			significance=0;
			objective_function_value=0;
		}
	}
	int_matrix_deallocation(resultant_equivalence_partition,number_of_class_labels);

	return objective_function_value;
}

void generate_equivalence_partition(int **rhepm_data_matrix,double *data_matrix,int *class_labels,int number_of_samples,int number_of_class_labels,double epsilon)
{	
	double *minimum_data_matrix,*maximum_data_matrix;
	double minimum,maximum;
	int *label;
	int i,j,k,l;
	
	minimum_data_matrix=(double *)malloc(sizeof(double)*number_of_class_labels);
	maximum_data_matrix=(double *)malloc(sizeof(double)*number_of_class_labels);
	label=(int *)malloc(sizeof(int)*number_of_class_labels);

	label[0]=class_labels[0];
	k=1;
	for(i=1;i<number_of_samples;i++)
	{
		l=0;
		for(j=0;j<k;j++)
			if(label[j]!=class_labels[i])
				l++;
		if(l==k)
		{
			label[k]=class_labels[i];
			k++;
		}
	}
	if(k!=number_of_class_labels)
	{
		printf("\nError: Error in Program.\n");
		exit(0);
	}
	minimum=data_matrix[0];
	maximum=data_matrix[0];
	for(i=1;i<number_of_samples;i++)
	{
		if(data_matrix[i]<minimum)
			minimum=data_matrix[i];
		if(data_matrix[i]>maximum)
			maximum=data_matrix[i];
	}
	for(i=0;i<number_of_class_labels;i++)
	{
		minimum_data_matrix[i]=maximum;
		maximum_data_matrix[i]=minimum;
		for(j=0;j<number_of_samples;j++)
		{
			if(class_labels[j]==label[i])
			{
				if(data_matrix[j]<minimum_data_matrix[i])
					minimum_data_matrix[i]=data_matrix[j];
				if(data_matrix[j]>maximum_data_matrix[i])
					maximum_data_matrix[i]=data_matrix[j];
			}
		}
	}
	for(i=0;i<number_of_class_labels;i++)
	{
		for(j=0;j<number_of_samples;j++)
		{
			rhepm_data_matrix[i][j]=0;
			if((data_matrix[j]>=minimum_data_matrix[i]-epsilon)&&(data_matrix[j]<=maximum_data_matrix[i]+epsilon))
				rhepm_data_matrix[i][j]=1;
		}
	}

	free(minimum_data_matrix);
	free(maximum_data_matrix);
	free(label);
}

double dependency_degree(int **rhepm_data_matrix,int number_of_samples,int number_of_class_labels)
{
	int *confusion_vector;
	int sum;
	int i,j;
	double gamma;

	confusion_vector=(int *)malloc(sizeof(int)*number_of_samples);

	for(j=0;j<number_of_samples;j++)
	{
		sum=0;
		for(i=0;i<number_of_class_labels;i++)
			sum+=rhepm_data_matrix[i][j];
		if(!sum)
		{
			printf("\nError: Error in RHEPM Computation.\n");
			exit(0);
		}
		else if(sum==1)
			confusion_vector[j]=0;
		else
			confusion_vector[j]=1;
	}
	sum=0;
	for(i=0;i<number_of_samples;i++)
		sum+=confusion_vector[i];
	gamma=1-(double)sum/number_of_samples;

	free(confusion_vector);

        return gamma;
}

double form_resultant_equivalence_partition_matrix(int **equivalence_partition1,int **equivalence_partition2,int **resultant_equivalence_partition,int number_of_class_labels,int number_of_samples)
{
	int i,j;

	for(i=0;i<number_of_class_labels;i++)
	{
		for(j=0;j<number_of_samples;j++)
		{
			resultant_equivalence_partition[i][j]=0;
			if(equivalence_partition1[i][j]&&equivalence_partition2[i][j])
				resultant_equivalence_partition[i][j]=1;
		}
	}
}

void write_output_file(char *filename,double **new_data_matrix,int *class_labels,int number_of_samples,int number_of_features,int number_of_class_labels)
{
	int i,j;
	FILE *fp_write;
	
	fp_write=fopen(filename,"w");
	if(fp_write==NULL)
	{
		printf("\nError: Error in Output File.\n");
		exit(0);
	}
	fprintf(fp_write,"%d\t%d\t%d\n",number_of_samples,number_of_features,number_of_class_labels);
	for(i=0;i<number_of_samples;i++)
	{
		for(j=0;j<number_of_features;j++)
			fprintf(fp_write,"%lf\t",new_data_matrix[i][j]);
		fprintf(fp_write,"%d\n",class_labels[i]);
	}
	fclose(fp_write);
}

void write_file(char *filename,double **new_data_matrix,int row,int column)
{
	int i,j;
	FILE *fp_write;
	
	fp_write=fopen(filename,"w");
	if(fp_write==NULL)
	{
		printf("\nError: Error in Output File.\n");
		exit(0);
	}
	for(i=0;i<row;i++)
	{
		for(j=0;j<column;j++)
			fprintf(fp_write,"%lf\t",new_data_matrix[i][j]);
		fprintf(fp_write,"\n");
	}
	fclose(fp_write);
}

void write_correlation_file(char *filename,double *new_data_matrix,int size_of_matrix)
{
	int i;
	FILE *fp_write;
	
	fp_write=fopen(filename,"w");
	if(fp_write==NULL)
	{
		printf("\nError: Error in Output File.\n");
		exit(0);
	}
	for(i=0;i<size_of_matrix;i++)
		fprintf(fp_write,"%lf\n",new_data_matrix[i]);
	fclose(fp_write);
}

void write_R_eigenvalue_eigenvector(char *data_filename,char *eigenvector_filename,char *eigenvalue_filename,char *filename)
{
	FILE *fp_write;
	
	fp_write=fopen(filename,"w");
	if(fp_write==NULL)
	{
		printf("\nError: Error in Output File.\n");
		exit(0);
	}
	fprintf(fp_write,"eigenvalue_eigenvector <- function(){");
	fprintf(fp_write,"\n\tx <- read.table('%s')",data_filename);
	fprintf(fp_write,"\n\tX <- as.matrix(x)");
	fprintf(fp_write,"\n\tev <- eigen(X)");
	fprintf(fp_write,"\n\twrite.table(Re(ev$vec), file='%s', sep='\\t', eol='\\n', row.names=FALSE, col.names=FALSE)",eigenvector_filename);
	fprintf(fp_write,"\n\twrite.table(Re(ev$val), file='%s', sep='\\t', eol='\\n', row.names=FALSE, col.names=FALSE)",eigenvalue_filename);
	fprintf(fp_write,"\n}\n");
	fprintf(fp_write,"\neigenvalue_eigenvector();");
	fclose(fp_write);
}

void double_matrix_deallocation(double **data,int row)
{
        int i;
        
        for(i=0;i<row;i++)
                free(data[i]);
        free(data);
}

void int_matrix_deallocation(int **data,int row)
{
        int i;
        
        for(i=0;i<row;i++)
                free(data[i]);
        free(data);
}
