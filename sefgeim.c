/****************************************************************************/
/*Source Code of SeFGeIM Algorithm written in C and R                       */
/*A. Mandal and P. Maji, SeFGeIM: Adaptive Generalized Multi-View Canonical */
/*Correlation Analysis for Incrementally Update Multiblock Data,            */
/*IEEE Transactions on Knowledge and Data Engineering, 2022.                */
/****************************************************************************/

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
	int number_of_samples;
	int number_of_features;
	int number_of_class_labels;
	int *class_labels;
	double **data_matrix;
};

struct rhepm_matrix
{
	int **data_matrix;
};

struct modality
{
	double **data_matrix;
};

struct featureset
{
	double *correlation;
	double *relevance;
	double *objective_function_value;
	struct modality canonical_variables_matrix;
	struct modality *basis_vector_matrix;
	struct rhepm_matrix *rhepm_data_matrix;
};

struct feature
{
	double relevance;
	double significance;
	double objective_function_value;
	int **rhepm_data_matrix;
};

void write_instruction(void);
char **char_matrix_allocation(int,int);
int **int_matrix_allocation(int,int);
double **double_matrix_allocation(int,int);
void preprocessing(char *,char *,int,int,double,double,double,double,double);
void read_filename(char **,char *,int);
struct dataset read_input_file(char *);
void read_file(char *,double **,int,int);
void read_eigenvector_file(char *,double **,int);
void read_eigenvalue_file(char *,double *,int);
void zero_mean(double **,double **,int,int);
void SeFGeIM(char *,struct modality *,int *,int *,int,int,int,int,double,double,double,double,double);
void between_set_covariance(double **,double **,double **,int,int,int);
void within_set_covariance(double **,double **,int,int);
void empirical_unbiased_between_set_variance(double **,double **,double **,double **,int,int,int);
void empirical_unbiased_within_set_variance(double **,double **,double **,int,int);
void matrix_transpose(double **,double **,int,int);
void matrix_multiplication(double **,double **,double **,int,int,int);
void eigenvalue_eigenvector(char *,double **,double **,double *,int);
int rhepm(struct featureset *,struct featureset,double **,int *,int *,int *,int,int,int,int,int,double,double);
void generate_equivalence_partition(int **,double *,int *,int,int,double,int);
double dependency_degree(int **,int,int);
double form_resultant_equivalence_partition_matrix(int **,int **,int **,int,int);
void write_output_file(char *,double **,int *,int,int,int);
void write_file(char *,double **,int,int);
void write_index_file(char *,int *,int);
void write_20_decimal_paces_file(char *,double **,int,int);
void write_correlation_file(char *,double *,int);
void write_lambda_file(char *,double **,int,int);
void write_R_eigenvalue_eigenvector(char *,char *,char *,char *);
void delete_file(char *,int,int);
void char_matrix_deallocation(char **,int);
void int_matrix_deallocation(int **,int);
void double_matrix_deallocation(double **,int);

int main(int argc,char *argv[])
{
	int o;
	int number_of_modalities,number_of_new_features;
	double lambda_minimum,lambda_maximum,delta;
	double epsilon,omega;
	extern char *optarg;
	char *modality_filename=NULL,*path;
	time_t t;
	struct timeb ti,tf;
	size_t allocsize=sizeof(char)*1024;

	number_of_modalities=2;
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

	while((o=getopt(argc,argv,"s:M:f:p:m:n:d:o:h"))!=EOF)
	{
		switch(o)
		{
			case 's': modality_filename=optarg;
				  printf("\tFile stem <%s>\n",modality_filename);
				  break;
			case 'M': number_of_modalities=atoi(optarg);
				  printf("\tNumber of Modalities: %d\n",number_of_modalities);
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
	if(modality_filename==NULL)
		write_instruction();
	(void)ftime(&ti);
	preprocessing(modality_filename,path,number_of_modalities,number_of_new_features,lambda_minimum,lambda_maximum,delta,epsilon,omega);
	(void)ftime(&tf);
	printf("\nTOTAL TIME REQUIRED for SeFGeIM=%d millisec\n",(int)(1000.0*(tf.time-ti.time)+(tf.millitm-ti.millitm)));
	printf("\n");
}

void write_instruction(void)
{
	system("clear");
	printf("s:\tInput Modality-name\n");
	printf("M:\tNumber of Modalities\n");
	printf("f:\tNumber of New Features\n");
	printf("p:\tPath for Input/Output Files\n");
	printf("m:\tMinimum Value of Regularization Parameter\n");
	printf("n:\tMaximum Value of Regularization Parameter\n");
	printf("d:\tIncrement of Regularization Parameter\n");
	printf("o:\tWeight Parameter\n");
	printf("h:\tHelp\n");
	exit(1);
}

char **char_matrix_allocation(int row,int column)
{
	char **data;
	int i;

	data=(char **)malloc(sizeof(char *)*row);
	assert(data!=NULL);
	for(i=0;i<row;i++)
	{
		data[i]=(char *)malloc(sizeof(char)*column);
		assert(data[i]!=NULL);
	}
	return data;
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

void preprocessing(char *modality_filename,char *path,int number_of_modalities,int number_of_new_features,double lambda_minimum,double lambda_maximum,double delta,double epsilon,double omega)
{
	int i,j;
	char *filename;
	int *number_of_features;
	char **modality_name;
	struct dataset temp_dataset;
	struct dataset *Dataset;
	struct modality *transpose_matrix,*zero_mean_matrix;
	struct stat st={0};

	filename=(char *)malloc(sizeof(char)*1000);
	number_of_features=(int *)malloc(sizeof(int)*number_of_modalities);
	modality_name=char_matrix_allocation(number_of_modalities,1000);
	Dataset=(struct dataset *)malloc(sizeof(struct dataset)*number_of_modalities);
	transpose_matrix=(struct modality *)malloc(sizeof(struct modality)*number_of_modalities);
	zero_mean_matrix=(struct modality *)malloc(sizeof(struct modality)*number_of_modalities);

	read_filename(modality_name,modality_filename,number_of_modalities);
	for(i=0;i<number_of_modalities;i++)
	{
		strcpy(filename,path);
		strcat(filename,modality_name[i]);
		Dataset[i]=read_input_file(filename);
		//Dataset[i]=read_input_file(modality_name[i]);
	}
	for(i=1;i<number_of_modalities;i++)
	{
		if(Dataset[i-1].number_of_samples!=Dataset[i].number_of_samples)
		{
			printf("\nError: Number of Samples in Dataset%d = %d\tNumber of Samples in Dataset%d = %d",i-1,Dataset[i-1].number_of_samples,i,Dataset[i].number_of_samples);
			exit(0);
		}
		for(j=0;j<Dataset[i].number_of_samples;j++)
		{
			if(Dataset[i-1].class_labels[j]!=Dataset[i].class_labels[j])
			{
				printf("\nError: Class labels%d[%d] = %d\tClass labels%d[%d] = %d",i-1,j+1,Dataset[i-1].class_labels[j],i,j+1,Dataset[i].class_labels[j]);
				exit(0);
			}
		}
	}
	for(i=0;i<number_of_modalities-1;i++)
		for (j=0;j<number_of_modalities-i-1;j++)
			if(Dataset[j].number_of_features>Dataset[j+1].number_of_features)
			{
				temp_dataset=Dataset[j];
				Dataset[j]=Dataset[j+1];
				Dataset[j+1]=temp_dataset; 
			}
	if((number_of_new_features>Dataset[0].number_of_features)||(!number_of_new_features))
		number_of_new_features=Dataset[0].number_of_features;
	for(i=0;i<number_of_modalities;i++)
	{
		number_of_features[i]=Dataset[i].number_of_features;
		transpose_matrix[i].data_matrix=double_matrix_allocation(Dataset[i].number_of_features,Dataset[i].number_of_samples);
		matrix_transpose(Dataset[i].data_matrix,transpose_matrix[i].data_matrix,Dataset[i].number_of_features,Dataset[i].number_of_samples);
		zero_mean_matrix[i].data_matrix=double_matrix_allocation(Dataset[i].number_of_features,Dataset[i].number_of_samples);
		zero_mean(transpose_matrix[i].data_matrix,zero_mean_matrix[i].data_matrix,Dataset[i].number_of_features,Dataset[i].number_of_samples);
	}
	strcpy(filename,path);
	strcat(filename,"SeFGeIM/");
	if(stat(filename,&st)==-1)
	    	mkdir(filename,0700);
	SeFGeIM(filename,zero_mean_matrix,Dataset[0].class_labels,number_of_features,Dataset[0].number_of_samples,number_of_new_features,Dataset[0].number_of_class_labels,number_of_modalities,lambda_minimum,lambda_maximum,delta,epsilon,omega);

	for(i=0;i<number_of_modalities;i++)
	{
		double_matrix_deallocation(Dataset[i].data_matrix,Dataset[i].number_of_samples);
		double_matrix_deallocation(transpose_matrix[i].data_matrix,Dataset[i].number_of_features);
		double_matrix_deallocation(zero_mean_matrix[i].data_matrix,Dataset[i].number_of_features);
		free(Dataset[i].class_labels);
	}
	free(Dataset);
	free(transpose_matrix);
	free(zero_mean_matrix);
	char_matrix_deallocation(modality_name,number_of_modalities);
	free(number_of_features);
	free(filename);
}

void read_filename(char **modality_name,char *modality_filename,int number_of_modalities)
{
	int i,j;
	FILE *fp_read;

	fp_read=fopen(modality_filename,"r");
	if(fp_read==NULL)
	{
		printf("\nError: Error in Input File.\n");
		exit(0);
	}
	for(i=0;i<number_of_modalities;i++)
	{
		j=0;
		do
		{
			modality_name[i][j]=getc(fp_read);
			j++;
		}while(modality_name[i][j-1]!='\n');
		modality_name[i][j-1]='\0';
	}
	fclose(fp_read);
}

struct dataset read_input_file(char *filename)
{
	int i,j;
	struct dataset Dataset;
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

void read_file(char *filename,double **data_matrix,int row,int column)
{
	int i,j;
	FILE *fp_read;

	fp_read=fopen(filename,"r");
	if(fp_read==NULL)
	{
		printf("\nError: Error in Input File1.\n");
		exit(0);
	}
	for(i=0;i<row;i++)
		for(j=0;j<column;j++)
			fscanf(fp_read,"%lf",&data_matrix[i][j]);
	fclose(fp_read);
}

void read_eigenvector_file(char *filename,double **data_matrix,int size_of_matrix)
{
	int i,j;
	FILE *fp_read; 

	fp_read=fopen(filename,"r");
	if(filename==NULL)
	{
		printf("\nError: Error in Input File2.\n");
		exit(0);
	}  
	for(i=0;i<size_of_matrix;i++)
		for(j=0;j<size_of_matrix;j++)
			fscanf(fp_read,"%lf",&data_matrix[i][j]);
	fclose(fp_read);
}

void read_eigenvalue_file(char *filename,double *data_matrix,int size_of_matrix)
{
	int i; 
	FILE *fp_read; 

	fp_read=fopen(filename,"r");
	if(filename==NULL)
	{
		printf("\nError: Error in Input File3.\n");
		exit(0);
	}  
	for(i=0;i<size_of_matrix;i++)
		fscanf(fp_read,"%lf",&data_matrix[i]);
	fclose(fp_read);
}

void zero_mean(double **data_matrix,double **new_data_matrix,int row,int column)
{
	int i,j;
	double sum;
	double *mean;

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

void SeFGeIM(char *path,struct modality *zero_mean_matrix,int *class_labels,int *number_of_features,int number_of_samples,int number_of_new_features,int number_of_class_labels,int number_of_modalities,double lambda_minimum,double lambda_maximum,double delta,double epsilon,double omega)
{
	int i,j,k,l,t,m,n;
	int count;
	int number_of_initial_modalities;
	int temp_index;
	int previous_number_of_features;
	int flag;
	double temp_eigenvalue;
	char *filename,*temp;
	int *index;
	int *sorted_number_of_features;
	int *combination_number;
	int *feature_from_modalities;
	int *new_combination_number;
	double *eigenvalue;
	double *correlation;
	int **temp_data_matrix;
	double **eigenvalue_of_covariance_data_matrix;
	double **inverse_covariance_data_matrix;
	double **temp_data_matrix1,**temp_data_matrix2,**temp_data_matrix3;
	double **h_data_matrix;
	double **eigenvector,**transpose_eigenvector;
	double **transpose_basis_vector_data_matrix;
	double **canonical_variables_data_matrix;
	double **previous_eigenvalue_eigenvector,**next_eigenvalue_eigenvector;
	double **theta_data_matrix,**transpose_theta_data_matrix;
	struct modality *eigenvector_of_covariance_data_matrix,*transpose_eigenvector_of_covariance_data_matrix;
	struct modality *sorted_inverse_covariance_matrix;
	struct modality *basis_vector;
	struct modality **covariance_matrix;
	struct modality **sorted_covariance_matrix;
	struct featureset final_optimal_featureset;
	struct featureset *optimal_featureset;

	count=(int)((lambda_maximum-lambda_minimum)/delta)+1;
	number_of_initial_modalities=2;
	filename=(char *)malloc(sizeof(char)*1000);
	temp=(char *)malloc(sizeof(char)*1000);
	eigenvalue_of_covariance_data_matrix=(double **)malloc(sizeof(double *)*number_of_modalities);
	eigenvector_of_covariance_data_matrix=(struct modality *)malloc(sizeof(struct modality)*number_of_modalities);
	transpose_eigenvector_of_covariance_data_matrix=(struct modality *)malloc(sizeof(struct modality)*number_of_modalities);
	covariance_matrix=(struct modality **)malloc(sizeof(struct modality *)*number_of_modalities);
	for(i=0;i<number_of_modalities;i++)
	{
		covariance_matrix[i]=(struct modality *)malloc(sizeof(struct modality)*number_of_modalities);
		for(j=0;j<number_of_modalities;j++)
		{
			covariance_matrix[i][j].data_matrix=double_matrix_allocation(number_of_features[i],number_of_features[j]);
			temp_data_matrix1=double_matrix_allocation(number_of_features[i],number_of_features[j]);
			if(i==j)
			{
				within_set_covariance(zero_mean_matrix[i].data_matrix,covariance_matrix[i][i].data_matrix,number_of_features[i],number_of_samples);
				empirical_unbiased_within_set_variance(zero_mean_matrix[i].data_matrix,covariance_matrix[i][i].data_matrix,temp_data_matrix1,number_of_features[i],number_of_samples);
				for(k=0;k<number_of_features[i];k++)
					for(l=0;l<number_of_features[i];l++)
						covariance_matrix[i][i].data_matrix[k][l]=temp_data_matrix1[k][l];
			}
			else if(i<j)
			{
				between_set_covariance(zero_mean_matrix[i].data_matrix,zero_mean_matrix[j].data_matrix,covariance_matrix[i][j].data_matrix,number_of_features[i],number_of_features[j],number_of_samples);
				empirical_unbiased_between_set_variance(zero_mean_matrix[i].data_matrix,zero_mean_matrix[j].data_matrix,covariance_matrix[i][j].data_matrix,temp_data_matrix1,number_of_features[i],number_of_features[j],number_of_samples);
				for(k=0;k<number_of_features[i];k++)
					for(l=0;l<number_of_features[j];l++)
						covariance_matrix[i][j].data_matrix[k][l]=temp_data_matrix1[k][l];
			}
			else
				matrix_transpose(covariance_matrix[j][i].data_matrix,covariance_matrix[i][j].data_matrix,number_of_features[i],number_of_features[j]);
		}
		eigenvalue_of_covariance_data_matrix[i]=(double *)malloc(sizeof(double)*number_of_features[i]);
		eigenvector_of_covariance_data_matrix[i].data_matrix=double_matrix_allocation(number_of_features[i],number_of_features[i]);
		eigenvalue_eigenvector(path,covariance_matrix[i][i].data_matrix,eigenvector_of_covariance_data_matrix[i].data_matrix,eigenvalue_of_covariance_data_matrix[i],number_of_features[i]);
		transpose_eigenvector_of_covariance_data_matrix[i].data_matrix=double_matrix_allocation(number_of_features[i],number_of_features[i]);
		matrix_transpose(eigenvector_of_covariance_data_matrix[i].data_matrix,transpose_eigenvector_of_covariance_data_matrix[i].data_matrix,number_of_features[i],number_of_features[i]);
		double_matrix_deallocation(temp_data_matrix1,number_of_features[i]);
	}
	for(t=0;t<count;t++)
	{
		for(i=0;i<number_of_modalities;i++)
		{
			inverse_covariance_data_matrix=double_matrix_allocation(number_of_features[i],number_of_features[i]);
			temp_data_matrix1=double_matrix_allocation(number_of_features[i],number_of_features[i]);
			for(j=0;j<number_of_features[i];j++)
				for(k=0;k<number_of_features[i];k++)
				{
					if((1/(eigenvalue_of_covariance_data_matrix[i][k]+t*delta))>=0.000001)
						temp_data_matrix1[j][k]=eigenvector_of_covariance_data_matrix[i].data_matrix[j][k]/(eigenvalue_of_covariance_data_matrix[i][k]+t*delta);
					else
						temp_data_matrix1[j][k]=eigenvector_of_covariance_data_matrix[i].data_matrix[j][k]/0.000001;
				}
			matrix_multiplication(temp_data_matrix1,transpose_eigenvector_of_covariance_data_matrix[i].data_matrix,inverse_covariance_data_matrix,number_of_features[i],number_of_features[i],number_of_features[i]);
			double_matrix_deallocation(temp_data_matrix1,number_of_features[i]);
			strcpy(filename,path);
			sprintf(temp,"inverse_covariance_data_matrix_%d_%d.txt",i+1,t+1);
			strcat(filename,temp);
			write_20_decimal_paces_file(filename,inverse_covariance_data_matrix,number_of_features[i],number_of_features[i]);
			double_matrix_deallocation(inverse_covariance_data_matrix,number_of_features[i]);
		}
	}
	for(i=0;i<number_of_modalities;i++)
	{
		double_matrix_deallocation(eigenvector_of_covariance_data_matrix[i].data_matrix,number_of_features[i]);
		double_matrix_deallocation(transpose_eigenvector_of_covariance_data_matrix[i].data_matrix,number_of_features[i]);
	}
	free(eigenvector_of_covariance_data_matrix);
	free(transpose_eigenvector_of_covariance_data_matrix);
	index=(int *)malloc(sizeof(int)*number_of_modalities);
	sorted_number_of_features=(int *)malloc(sizeof(int)*number_of_modalities);
	eigenvalue=(double *)malloc(sizeof(double)*number_of_modalities);
	for(i=0;i<number_of_modalities;i++)
	{
		index[i]=i;
		eigenvalue[i]=eigenvalue_of_covariance_data_matrix[i][0];
	}
	for(i=0;i<number_of_modalities;i++)
		free(eigenvalue_of_covariance_data_matrix[i]);
	free(eigenvalue_of_covariance_data_matrix);
	for(i=0;i<number_of_modalities-1;i++)
		for(j=0;j<number_of_modalities-i-1;j++)
			if(eigenvalue[j]<eigenvalue[j+1])
			{
				temp_eigenvalue=eigenvalue[j];
				eigenvalue[j]=eigenvalue[j+1];
				eigenvalue[j+1]=temp_eigenvalue;
				temp_index=index[j];
				index[j]=index[j+1];
				index[j+1]=temp_index;
			}
	free(eigenvalue);
	if(number_of_features[index[0]]>number_of_features[index[1]])
	{
		temp_index=index[0];
		index[0]=index[1];
		index[1]=temp_index;
	}
	strcpy(filename,path);
	strcat(filename,"sorted_index.txt");
	write_index_file(filename,index,number_of_modalities);
	sorted_covariance_matrix=(struct modality **)malloc(sizeof(struct modality *)*number_of_modalities);
	sorted_inverse_covariance_matrix=(struct modality *)malloc(sizeof(struct modality)*number_of_modalities);
	for(i=0;i<number_of_modalities;i++)
	{
		sorted_covariance_matrix[i]=(struct modality *)malloc(sizeof(struct modality)*number_of_modalities);
		for(j=0;j<number_of_modalities;j++)
			sorted_covariance_matrix[i][j].data_matrix=double_matrix_allocation(number_of_features[index[i]],number_of_features[index[j]]);
		sorted_inverse_covariance_matrix[i].data_matrix=double_matrix_allocation(number_of_features[index[i]],number_of_features[index[i]]);
	}
	for(i=0;i<number_of_modalities;i++)
	{
		sorted_number_of_features[i]=number_of_features[index[i]];
		for(j=0;j<number_of_modalities;j++)
			for(k=0;k<number_of_features[index[i]];k++)
				for(l=0;l<number_of_features[index[j]];l++)
					sorted_covariance_matrix[i][j].data_matrix[k][l]=covariance_matrix[index[i]][index[j]].data_matrix[k][l];
	}
	for(i=0;i<number_of_modalities;i++)
	{
		for(j=0;j<number_of_modalities;j++)
			double_matrix_deallocation(covariance_matrix[i][j].data_matrix,number_of_features[i]);
		free(covariance_matrix[i]);
	}
	free(covariance_matrix);
	for(i=0;i<count;i++)
	{
		strcpy(filename,path);
		sprintf(temp,"inverse_covariance_data_matrix_%d_%d.txt",index[0]+1,i+1);
		strcat(filename,temp);
		read_file(filename,sorted_inverse_covariance_matrix[0].data_matrix,sorted_number_of_features[0],sorted_number_of_features[0]);
		for(j=0;j<count;j++)
		{
			strcpy(filename,path);
			sprintf(temp,"inverse_covariance_data_matrix_%d_%d.txt",index[1]+1,j+1);
			strcat(filename,temp);
			read_file(filename,sorted_inverse_covariance_matrix[1].data_matrix,sorted_number_of_features[1],sorted_number_of_features[1]);
			h_data_matrix=double_matrix_allocation(sorted_number_of_features[0],sorted_number_of_features[0]);
			temp_data_matrix1=double_matrix_allocation(sorted_number_of_features[0],sorted_number_of_features[1]);
			temp_data_matrix2=double_matrix_allocation(sorted_number_of_features[0],sorted_number_of_features[1]);
			matrix_multiplication(sorted_inverse_covariance_matrix[0].data_matrix,sorted_covariance_matrix[0][1].data_matrix,temp_data_matrix1,sorted_number_of_features[0],sorted_number_of_features[0],sorted_number_of_features[1]);
			matrix_multiplication(temp_data_matrix1,sorted_inverse_covariance_matrix[1].data_matrix,temp_data_matrix2,sorted_number_of_features[0],sorted_number_of_features[1],sorted_number_of_features[1]);
			matrix_multiplication(temp_data_matrix2,sorted_covariance_matrix[1][0].data_matrix,h_data_matrix,sorted_number_of_features[0],sorted_number_of_features[1],sorted_number_of_features[0]);
			double_matrix_deallocation(temp_data_matrix1,sorted_number_of_features[0]);
			double_matrix_deallocation(temp_data_matrix2,sorted_number_of_features[0]);
			eigenvalue=(double *)malloc(sizeof(double)*sorted_number_of_features[0]);
			eigenvector=double_matrix_allocation(sorted_number_of_features[0],sorted_number_of_features[0]);
			eigenvalue_eigenvector(path,h_data_matrix,eigenvector,eigenvalue,sorted_number_of_features[0]);
			double_matrix_deallocation(h_data_matrix,sorted_number_of_features[0]);
			temp_data_matrix1=double_matrix_allocation(sorted_number_of_features[0],number_of_new_features);
			temp_data_matrix2=double_matrix_allocation(sorted_number_of_features[1],number_of_new_features);
			temp_data_matrix3=double_matrix_allocation(sorted_number_of_features[1],sorted_number_of_features[0]);
			correlation=(double *)malloc(sizeof(double)*number_of_new_features);
			for(k=0;k<sorted_number_of_features[0];k++)
				for(l=0;l<number_of_new_features;l++)
					temp_data_matrix1[k][l]=eigenvector[k][l];
			for(k=0;k<number_of_new_features;k++)
				correlation[k]=sqrt(fabs(eigenvalue[k]));
			double_matrix_deallocation(eigenvector,sorted_number_of_features[0]);
			free(eigenvalue);
			matrix_multiplication(sorted_inverse_covariance_matrix[1].data_matrix,sorted_covariance_matrix[1][0].data_matrix,temp_data_matrix3,sorted_number_of_features[1],sorted_number_of_features[1],sorted_number_of_features[0]);
			matrix_multiplication(temp_data_matrix3,temp_data_matrix1,temp_data_matrix2,sorted_number_of_features[1],sorted_number_of_features[0],number_of_new_features);
			strcpy(filename,path);
			sprintf(temp,"eigenvector1_%d_%d.txt",i+1,j+1);
			strcat(filename,temp);
			write_20_decimal_paces_file(filename,temp_data_matrix1,sorted_number_of_features[0],number_of_new_features);
			strcpy(filename,path);
			sprintf(temp,"eigenvector2_%d_%d.txt",i+1,j+1);
			strcat(filename,temp);
			write_20_decimal_paces_file(filename,temp_data_matrix2,sorted_number_of_features[1],number_of_new_features);
			strcpy(filename,path);
			sprintf(temp,"correlation_%d_%d.txt",i+1,j+1);
			strcat(filename,temp);
			write_correlation_file(filename,correlation,number_of_new_features);
			double_matrix_deallocation(temp_data_matrix3,sorted_number_of_features[1]);
			free(correlation);
			canonical_variables_data_matrix=double_matrix_allocation(number_of_new_features,number_of_samples);
			for(k=0;k<number_of_new_features;k++)
				for(l=0;l<number_of_samples;l++)
					canonical_variables_data_matrix[k][l]=0;
			transpose_basis_vector_data_matrix=double_matrix_allocation(number_of_new_features,sorted_number_of_features[0]);
			temp_data_matrix3=double_matrix_allocation(number_of_new_features,number_of_samples);
			matrix_transpose(temp_data_matrix1,transpose_basis_vector_data_matrix,number_of_new_features,sorted_number_of_features[0]);
			matrix_multiplication(transpose_basis_vector_data_matrix,zero_mean_matrix[index[0]].data_matrix,temp_data_matrix3,number_of_new_features,sorted_number_of_features[0],number_of_samples);
			for(k=0;k<number_of_new_features;k++)
				for(l=0;l<number_of_samples;l++)
					canonical_variables_data_matrix[k][l]+=temp_data_matrix3[k][l];
			double_matrix_deallocation(transpose_basis_vector_data_matrix,number_of_new_features);
			transpose_basis_vector_data_matrix=double_matrix_allocation(number_of_new_features,sorted_number_of_features[1]);
			matrix_transpose(temp_data_matrix2,transpose_basis_vector_data_matrix,number_of_new_features,sorted_number_of_features[1]);
			matrix_multiplication(transpose_basis_vector_data_matrix,zero_mean_matrix[index[1]].data_matrix,temp_data_matrix3,number_of_new_features,sorted_number_of_features[1],number_of_samples);
			for(k=0;k<number_of_new_features;k++)
				for(l=0;l<number_of_samples;l++)
					canonical_variables_data_matrix[k][l]+=temp_data_matrix3[k][l];
			for(k=0;k<number_of_new_features;k++)
				for(l=0;l<number_of_samples;l++)
					canonical_variables_data_matrix[k][l]/=2;
			double_matrix_deallocation(transpose_basis_vector_data_matrix,number_of_new_features);
			double_matrix_deallocation(temp_data_matrix1,sorted_number_of_features[0]);
			double_matrix_deallocation(temp_data_matrix2,sorted_number_of_features[1]);
			double_matrix_deallocation(temp_data_matrix3,number_of_new_features);
			strcpy(filename,path);
			sprintf(temp,"canonical_variables_%d_%d.txt",i+1,j+1);
			strcat(filename,temp);
			write_20_decimal_paces_file(filename,canonical_variables_data_matrix,number_of_new_features,number_of_samples);
			double_matrix_deallocation(canonical_variables_data_matrix,number_of_new_features);
			printf("\nIteration=%d\t%d",i+1,j+1);
		}
	}
	combination_number=(int *)malloc(sizeof(int)*number_of_new_features);
	feature_from_modalities=(int *)malloc(sizeof(int)*number_of_new_features);
	temp_data_matrix=int_matrix_allocation(count*count,number_of_initial_modalities);
	optimal_featureset=(struct featureset *)malloc(sizeof(struct featureset)*number_of_modalities-1);
	for(i=0;i<number_of_modalities-1;i++)
	{
		optimal_featureset[i].correlation=(double *)malloc(sizeof(double)*number_of_new_features);
		optimal_featureset[i].relevance=(double *)malloc(sizeof(double)*number_of_new_features);
		optimal_featureset[i].objective_function_value=(double *)malloc(sizeof(double)*number_of_new_features);
		optimal_featureset[i].canonical_variables_matrix.data_matrix=double_matrix_allocation(number_of_samples,number_of_new_features);
		optimal_featureset[i].basis_vector_matrix=(struct modality *)malloc(sizeof(struct modality)*number_of_modalities);
		optimal_featureset[i].rhepm_data_matrix=(struct rhepm_matrix *)malloc(sizeof(struct rhepm_matrix)*number_of_new_features);
	}
	for(i=0;i<number_of_modalities-1;i++)
	{
		for(j=0;j<number_of_modalities;j++)
			optimal_featureset[i].basis_vector_matrix[j].data_matrix=double_matrix_allocation(sorted_number_of_features[j],number_of_new_features);
		for(j=0;j<number_of_new_features;j++)
			optimal_featureset[i].rhepm_data_matrix[j].data_matrix=int_matrix_allocation(number_of_class_labels,number_of_samples);
	}
	final_optimal_featureset.correlation=(double *)malloc(sizeof(double)*number_of_new_features);
	final_optimal_featureset.relevance=(double *)malloc(sizeof(double)*number_of_new_features);
	final_optimal_featureset.objective_function_value=(double *)malloc(sizeof(double)*number_of_new_features);
	final_optimal_featureset.canonical_variables_matrix.data_matrix=double_matrix_allocation(number_of_samples,number_of_new_features);
	final_optimal_featureset.basis_vector_matrix=(struct modality *)malloc(sizeof(struct modality)*number_of_modalities);
	final_optimal_featureset.rhepm_data_matrix=(struct rhepm_matrix *)malloc(sizeof(struct rhepm_matrix)*number_of_new_features);
	for(i=0;i<number_of_samples;i++)
		for(j=0;j<number_of_new_features;j++)
			final_optimal_featureset.canonical_variables_matrix.data_matrix[i][j]=0;
	for(i=0;i<number_of_modalities;i++)
	{
		final_optimal_featureset.basis_vector_matrix[i].data_matrix=double_matrix_allocation(sorted_number_of_features[i],number_of_new_features);
		for(j=0;j<sorted_number_of_features[i];j++)
			for(k=0;k<number_of_new_features;k++)
				final_optimal_featureset.basis_vector_matrix[i].data_matrix[j][k]=0;
	}  
	for(i=0;i<number_of_new_features;i++)
	{
		final_optimal_featureset.rhepm_data_matrix[i].data_matrix=int_matrix_allocation(number_of_class_labels,number_of_samples);
		for(j=0;j<number_of_class_labels;j++)
			for(k=0;k<number_of_samples;k++)
				final_optimal_featureset.rhepm_data_matrix[i].data_matrix[j][k]=0;
	}  
	for(t=0;t<number_of_new_features;t++)
	{
		l=0;
		temp_data_matrix1=double_matrix_allocation(number_of_samples,count*count);
		for(i=0;i<count;i++)
			for(j=0;j<count;j++)
			{
				strcpy(filename,path);
				sprintf(temp,"canonical_variables_%d_%d.txt",i+1,j+1);
				strcat(filename,temp);
				temp_data_matrix2=double_matrix_allocation(number_of_new_features,number_of_samples);
				read_file(filename,temp_data_matrix2,number_of_new_features,number_of_samples);
				for(k=0;k<number_of_samples;k++)
					temp_data_matrix1[k][l]=temp_data_matrix2[t][k];
				double_matrix_deallocation(temp_data_matrix2,number_of_new_features);
				temp_data_matrix[l][0]=i;
				temp_data_matrix[l][1]=j;
				l++;
			}
		if(l!=count*count)
		{
			printf("\nError: Error in Canonical Variables.\n");
			exit(0);
		} 
		flag=0;
		flag=rhepm(optimal_featureset,final_optimal_featureset,temp_data_matrix1,class_labels,combination_number,feature_from_modalities,2,number_of_samples,t,l,number_of_class_labels,epsilon,omega);
		double_matrix_deallocation(temp_data_matrix1,number_of_samples);
		strcpy(filename,path);
		sprintf(temp,"eigenvector1_%d_%d.txt",temp_data_matrix[combination_number[t]][0]+1,temp_data_matrix[combination_number[t]][1]+1);
		strcat(filename,temp);
		temp_data_matrix1=double_matrix_allocation(sorted_number_of_features[0],number_of_new_features);
		read_file(filename,temp_data_matrix1,sorted_number_of_features[0],number_of_new_features);
		for(i=0;i<sorted_number_of_features[0];i++)
		{
			optimal_featureset[0].basis_vector_matrix[0].data_matrix[i][t]=temp_data_matrix1[i][t];
			final_optimal_featureset.basis_vector_matrix[0].data_matrix[i][t]=temp_data_matrix1[i][t];
		}
		double_matrix_deallocation(temp_data_matrix1,sorted_number_of_features[0]);
		strcpy(filename,path);
		sprintf(temp,"eigenvector2_%d_%d.txt",temp_data_matrix[combination_number[t]][0]+1,temp_data_matrix[combination_number[t]][1]+1);
		strcat(filename,temp);
		temp_data_matrix1=double_matrix_allocation(sorted_number_of_features[1],number_of_new_features);
		read_file(filename,temp_data_matrix1,sorted_number_of_features[1],number_of_new_features);
		for(i=0;i<sorted_number_of_features[1];i++)
		{
			optimal_featureset[0].basis_vector_matrix[1].data_matrix[i][t]=temp_data_matrix1[i][t];
			final_optimal_featureset.basis_vector_matrix[1].data_matrix[i][t]=temp_data_matrix1[i][t];
		}
		double_matrix_deallocation(temp_data_matrix1,sorted_number_of_features[1]);
		strcpy(filename,path);
		sprintf(temp,"canonical_variables_%d_%d.txt",temp_data_matrix[combination_number[t]][0]+1,temp_data_matrix[combination_number[t]][1]+1);
		strcat(filename,temp);
		temp_data_matrix1=double_matrix_allocation(number_of_new_features,number_of_samples);
		read_file(filename,temp_data_matrix1,number_of_new_features,number_of_samples);
		for(i=0;i<number_of_samples;i++)
		{
			optimal_featureset[0].canonical_variables_matrix.data_matrix[i][t]=temp_data_matrix1[t][i];
			final_optimal_featureset.canonical_variables_matrix.data_matrix[i][t]=temp_data_matrix1[t][i];
		}
		double_matrix_deallocation(temp_data_matrix1,number_of_new_features);
		strcpy(filename,path);
		sprintf(temp,"correlation_%d_%d.txt",temp_data_matrix[combination_number[t]][0]+1,temp_data_matrix[combination_number[t]][1]+1);
		strcat(filename,temp);
		correlation=(double *)malloc(sizeof(double)*number_of_new_features);
		read_eigenvalue_file(filename,correlation,number_of_new_features);
		optimal_featureset[0].correlation[t]=correlation[t];
		final_optimal_featureset.correlation[t]=correlation[t];
		free(correlation);
	}
	for(i=0;i<number_of_new_features;i++)
		feature_from_modalities[i]=2;
	printf("\nModality Complete=1\nModality Complete=2");
	basis_vector=(struct modality *)malloc(sizeof(struct modality)*number_of_modalities);
	for(i=0;i<number_of_modalities;i++)
		basis_vector[i].data_matrix=double_matrix_allocation(sorted_number_of_features[i],number_of_new_features);
	new_combination_number=(int *)malloc(sizeof(int)*number_of_new_features);
	for(i=number_of_initial_modalities;i<number_of_modalities;i++)
	{
		previous_number_of_features=0;
		for(j=0;j<i;j++)
			previous_number_of_features+=sorted_number_of_features[j];
		temp_data_matrix1=double_matrix_allocation(previous_number_of_features,number_of_new_features);
		temp_data_matrix2=double_matrix_allocation(number_of_new_features,previous_number_of_features);
		previous_eigenvalue_eigenvector=double_matrix_allocation(previous_number_of_features,previous_number_of_features);
		t=0;
		for(j=0;j<i;j++)
			for(k=0;k<sorted_number_of_features[j];k++)
			{
				for(l=0;l<number_of_new_features;l++)
					temp_data_matrix1[t][l]=optimal_featureset[0].basis_vector_matrix[j].data_matrix[k][l];
				t++;
			}
		if(t!=previous_number_of_features)
		{
			printf("\nError: Error in Previous Eigenvalue Eigenvector.\n");
			exit(0);
		}  
		matrix_transpose(temp_data_matrix1,temp_data_matrix2,number_of_new_features,previous_number_of_features);
		for(j=0;j<previous_number_of_features;j++)
			for(k=0;k<number_of_new_features;k++)
				temp_data_matrix1[j][k]*=pow(optimal_featureset[0].correlation[k],2);
		matrix_multiplication(temp_data_matrix1,temp_data_matrix2,previous_eigenvalue_eigenvector,previous_number_of_features,number_of_new_features,previous_number_of_features);
		double_matrix_deallocation(temp_data_matrix1,previous_number_of_features);
		double_matrix_deallocation(temp_data_matrix2,number_of_new_features);
		for(m=0;m<number_of_new_features;m++)
		{
			for(j=0;j<i;j++)
			{
				if(j<number_of_initial_modalities)
				{
					strcpy(filename,path);
					sprintf(temp,"inverse_covariance_data_matrix_%d_%d.txt",j+1,temp_data_matrix[combination_number[m]][j]+1);
					strcat(filename,temp);
					read_file(filename,sorted_inverse_covariance_matrix[j].data_matrix,sorted_number_of_features[j],sorted_number_of_features[j]);
				}
				else
				{
					strcpy(filename,path);
					sprintf(temp,"inverse_covariance_data_matrix_%d_%d.txt",j+1,new_combination_number[m]+1);
					strcat(filename,temp);
					read_file(filename,sorted_inverse_covariance_matrix[j].data_matrix,sorted_number_of_features[j],sorted_number_of_features[j]);
				}
			}
			if(!m)
				for(n=0;n<count;n++)
				{
					strcpy(filename,path);
					sprintf(temp,"inverse_covariance_data_matrix_%d_%d.txt",i+1,n+1);
					strcat(filename,temp);
					read_file(filename,sorted_inverse_covariance_matrix[i].data_matrix,sorted_number_of_features[i],sorted_number_of_features[i]);
					h_data_matrix=double_matrix_allocation(sorted_number_of_features[i],sorted_number_of_features[i]);
					for(j=0;j<sorted_number_of_features[i];j++)
						for(k=0;k<sorted_number_of_features[i];k++)
							h_data_matrix[j][k]=0;
					for(j=0;j<i;j++)
					{
						temp_data_matrix1=double_matrix_allocation(sorted_number_of_features[i],sorted_number_of_features[j]);
						temp_data_matrix2=double_matrix_allocation(sorted_number_of_features[i],sorted_number_of_features[j]);
						temp_data_matrix3=double_matrix_allocation(sorted_number_of_features[i],sorted_number_of_features[i]);
						matrix_multiplication(sorted_inverse_covariance_matrix[i].data_matrix,sorted_covariance_matrix[i][j].data_matrix,temp_data_matrix1,sorted_number_of_features[i],sorted_number_of_features[i],sorted_number_of_features[j]);
						matrix_multiplication(temp_data_matrix1,sorted_inverse_covariance_matrix[j].data_matrix,temp_data_matrix2,sorted_number_of_features[i],sorted_number_of_features[j],sorted_number_of_features[j]);
						matrix_multiplication(temp_data_matrix2,sorted_covariance_matrix[j][i].data_matrix,temp_data_matrix3,sorted_number_of_features[i],sorted_number_of_features[j],sorted_number_of_features[i]);
						for(k=0;k<sorted_number_of_features[i];k++)
							for(l=0;l<sorted_number_of_features[i];l++)
								h_data_matrix[k][l]+=temp_data_matrix3[k][l];
						double_matrix_deallocation(temp_data_matrix1,sorted_number_of_features[i]);
						double_matrix_deallocation(temp_data_matrix2,sorted_number_of_features[i]);
						double_matrix_deallocation(temp_data_matrix3,sorted_number_of_features[i]);
					}
					eigenvalue=(double *)malloc(sizeof(double)*sorted_number_of_features[i]);
					eigenvector=double_matrix_allocation(sorted_number_of_features[i],sorted_number_of_features[i]);
					eigenvalue_eigenvector(path,h_data_matrix,eigenvector,eigenvalue,sorted_number_of_features[i]);
					double_matrix_deallocation(h_data_matrix,sorted_number_of_features[i]);
					correlation=(double *)malloc(sizeof(double)*number_of_new_features);
					for(j=0;j<sorted_number_of_features[i];j++)
						for(k=0;k<number_of_new_features;k++)
							basis_vector[i].data_matrix[j][k]=eigenvector[j][k];
					for(j=0;j<number_of_new_features;j++)
						correlation[j]=sqrt(fabs(eigenvalue[j]));
					double_matrix_deallocation(eigenvector,sorted_number_of_features[i]);
					free(eigenvalue);
					strcpy(filename,path);
					sprintf(temp,"eigenvector%d_%d.txt",i+1,n+1);
					strcat(filename,temp);
					write_20_decimal_paces_file(filename,basis_vector[i].data_matrix,sorted_number_of_features[i],number_of_new_features);
					strcpy(filename,path);
					sprintf(temp,"correlation_%d.txt",n+1);
					strcat(filename,temp);
					write_correlation_file(filename,correlation,number_of_new_features);
					transpose_eigenvector=double_matrix_allocation(number_of_new_features,sorted_number_of_features[i]);
					theta_data_matrix=double_matrix_allocation(previous_number_of_features,sorted_number_of_features[i]);
					transpose_theta_data_matrix=double_matrix_allocation(sorted_number_of_features[i],previous_number_of_features);
					t=0;
					for(j=0;j<i;j++)
					{
						temp_data_matrix1=double_matrix_allocation(sorted_number_of_features[j],sorted_number_of_features[i]);
						matrix_multiplication(sorted_inverse_covariance_matrix[j].data_matrix,sorted_covariance_matrix[j][i].data_matrix,temp_data_matrix1,sorted_number_of_features[j],sorted_number_of_features[j],sorted_number_of_features[i]);
						for(k=0;k<sorted_number_of_features[j];k++)
						{
							for(l=0;l<sorted_number_of_features[i];l++)
								theta_data_matrix[t][l]=temp_data_matrix1[k][l];
							t++;
						}
						double_matrix_deallocation(temp_data_matrix1,sorted_number_of_features[j]);
					}
					if(t!=previous_number_of_features)
					{
						printf("\nError: Error in Theta Matrix.\n");
						exit(0);
					}
					matrix_transpose(basis_vector[i].data_matrix,transpose_eigenvector,number_of_new_features,sorted_number_of_features[i]);
					matrix_transpose(theta_data_matrix,transpose_theta_data_matrix,sorted_number_of_features[i],previous_number_of_features);
					temp_data_matrix1=double_matrix_allocation(sorted_number_of_features[i],number_of_new_features);
					temp_data_matrix2=double_matrix_allocation(previous_number_of_features,number_of_new_features);
					temp_data_matrix3=double_matrix_allocation(previous_number_of_features,sorted_number_of_features[i]);
					next_eigenvalue_eigenvector=double_matrix_allocation(previous_number_of_features,previous_number_of_features);
					for(j=0;j<sorted_number_of_features[i];j++)
						for(k=0;k<number_of_new_features;k++)
							temp_data_matrix1[j][k]=basis_vector[i].data_matrix[j][k]*pow(correlation[k],2);
					matrix_multiplication(theta_data_matrix,temp_data_matrix1,temp_data_matrix2,previous_number_of_features,sorted_number_of_features[i],number_of_new_features);
					matrix_multiplication(temp_data_matrix2,transpose_eigenvector,temp_data_matrix3,previous_number_of_features,number_of_new_features,sorted_number_of_features[i]);
					matrix_multiplication(temp_data_matrix3,transpose_theta_data_matrix,next_eigenvalue_eigenvector,previous_number_of_features,sorted_number_of_features[i],previous_number_of_features);
					double_matrix_deallocation(temp_data_matrix1,sorted_number_of_features[i]);
					double_matrix_deallocation(temp_data_matrix2,previous_number_of_features);
					double_matrix_deallocation(temp_data_matrix3,previous_number_of_features);
					double_matrix_deallocation(transpose_eigenvector,number_of_new_features);
					double_matrix_deallocation(theta_data_matrix,previous_number_of_features);
					double_matrix_deallocation(transpose_theta_data_matrix,sorted_number_of_features[i]);
					free(correlation);
					h_data_matrix=double_matrix_allocation(previous_number_of_features,previous_number_of_features);
					for(j=0;j<previous_number_of_features;j++)
						for(k=0;k<previous_number_of_features;k++)
							h_data_matrix[j][k]=previous_eigenvalue_eigenvector[j][k]+next_eigenvalue_eigenvector[j][k];
					double_matrix_deallocation(next_eigenvalue_eigenvector,previous_number_of_features);
					eigenvalue=(double *)malloc(sizeof(double)*previous_number_of_features);
					eigenvector=double_matrix_allocation(previous_number_of_features,previous_number_of_features);
					eigenvalue_eigenvector(path,h_data_matrix,eigenvector,eigenvalue,previous_number_of_features);
					double_matrix_deallocation(h_data_matrix,previous_number_of_features);
					t=0;
					for(j=0;j<i;j++)
					{
						for(k=0;k<sorted_number_of_features[j];k++)
						{
							for(l=0;l<number_of_new_features;l++)
								basis_vector[j].data_matrix[k][l]=eigenvector[t][l];
							t++;
						}
						strcpy(filename,path);
						sprintf(temp,"eigenvector%d_%d.txt",j+1,n+1);
						strcat(filename,temp);
						write_20_decimal_paces_file(filename,basis_vector[j].data_matrix,sorted_number_of_features[j],number_of_new_features);
					}
					if(t!=previous_number_of_features)
					{
						printf("\nError: Error in Next Eigenvalue Eigenvector.\n");
						exit(0);
					}
					double_matrix_deallocation(eigenvector,previous_number_of_features);
					free(eigenvalue);
					printf("\nIteration=%d",n+1);
					canonical_variables_data_matrix=double_matrix_allocation(number_of_new_features,number_of_samples);
					temp_data_matrix1=double_matrix_allocation(number_of_new_features,number_of_samples);
					for(j=0;j<number_of_new_features;j++)
						for(k=0;k<number_of_samples;k++)
							canonical_variables_data_matrix[j][k]=0;
					for(j=0;j<i+1;j++)
					{
						transpose_basis_vector_data_matrix=double_matrix_allocation(number_of_new_features,sorted_number_of_features[j]);
						matrix_transpose(basis_vector[j].data_matrix,transpose_basis_vector_data_matrix,number_of_new_features,sorted_number_of_features[j]);
						matrix_multiplication(transpose_basis_vector_data_matrix,zero_mean_matrix[index[j]].data_matrix,temp_data_matrix1,number_of_new_features,sorted_number_of_features[j],number_of_samples);
						for(k=0;k<number_of_new_features;k++)
							for(l=0;l<number_of_samples;l++)
								canonical_variables_data_matrix[k][l]+=temp_data_matrix1[k][l];
						double_matrix_deallocation(transpose_basis_vector_data_matrix,number_of_new_features);
					}
					for(j=0;j<number_of_new_features;j++)
						for(k=0;k<number_of_samples;k++)
							canonical_variables_data_matrix[j][k]/=(i+1);
					strcpy(filename,path);
					sprintf(temp,"canonical_variables_%d.txt",n+1);
					strcat(filename,temp);
					write_20_decimal_paces_file(filename,canonical_variables_data_matrix,number_of_new_features,number_of_samples);
					double_matrix_deallocation(canonical_variables_data_matrix,number_of_new_features);
					double_matrix_deallocation(temp_data_matrix1,number_of_new_features);
				}
			temp_data_matrix1=double_matrix_allocation(number_of_samples,count);
			for(n=0;n<count;n++)
			{
				strcpy(filename,path);
				sprintf(temp,"canonical_variables_%d.txt",n+1);
				strcat(filename,temp);
				temp_data_matrix2=double_matrix_allocation(number_of_new_features,number_of_samples);
				read_file(filename,temp_data_matrix2,number_of_new_features,number_of_samples);
				for(k=0;k<number_of_samples;k++)
					temp_data_matrix1[k][n]=temp_data_matrix2[m][k];
				double_matrix_deallocation(temp_data_matrix2,number_of_new_features);
			}
			flag=0;
			flag=rhepm(optimal_featureset,final_optimal_featureset,temp_data_matrix1,class_labels,new_combination_number,feature_from_modalities,i+1,number_of_samples,m,count,number_of_class_labels,epsilon,omega);
			double_matrix_deallocation(temp_data_matrix1,number_of_samples);
			for(j=0;j<i+1;j++)
			{
				strcpy(filename,path);
				sprintf(temp,"eigenvector%d_%d.txt",j+1,new_combination_number[m]+1);
				strcat(filename,temp);
				temp_data_matrix1=double_matrix_allocation(sorted_number_of_features[j],number_of_new_features);
				read_file(filename,temp_data_matrix1,sorted_number_of_features[j],number_of_new_features);
				for(k=0;k<sorted_number_of_features[j];k++)
				{
					optimal_featureset[i-1].basis_vector_matrix[j].data_matrix[k][m]=temp_data_matrix1[k][m];
					if(!flag)
						final_optimal_featureset.basis_vector_matrix[j].data_matrix[k][m]=temp_data_matrix1[k][m];
				}
				double_matrix_deallocation(temp_data_matrix1,sorted_number_of_features[j]);
			}
			strcpy(filename,path);
			sprintf(temp,"canonical_variables_%d.txt",new_combination_number[m]+1);
			strcat(filename,temp);
			temp_data_matrix1=double_matrix_allocation(number_of_new_features,number_of_samples);
			read_file(filename,temp_data_matrix1,number_of_new_features,number_of_samples);
			for(j=0;j<number_of_samples;j++)
			{
				optimal_featureset[i-1].canonical_variables_matrix.data_matrix[j][m]=temp_data_matrix1[m][j];
				if(!flag)
					final_optimal_featureset.canonical_variables_matrix.data_matrix[j][m]=temp_data_matrix1[m][j];
			}
			double_matrix_deallocation(temp_data_matrix1,number_of_new_features);
			strcpy(filename,path);
			sprintf(temp,"correlation_%d.txt",new_combination_number[m]+1);
			strcat(filename,temp);
			correlation=(double *)malloc(sizeof(double)*number_of_new_features);
			read_eigenvalue_file(filename,correlation,number_of_new_features);
			optimal_featureset[i-1].correlation[m]=correlation[m];
			if(!flag)
				final_optimal_featureset.correlation[m]=correlation[m];
			free(correlation);
			printf("\nFeature=%d\tModality=%d",m+1,i+1);
		}
		double_matrix_deallocation(previous_eigenvalue_eigenvector,previous_number_of_features);
		printf("\nModality Complete=%d",i+1);
	}
	for(i=0;i<number_of_modalities;i++)
	{
		strcpy(filename,path);
		sprintf(temp,"basis_vector%d.txt",i+1);
		strcat(filename,temp);
		write_20_decimal_paces_file(filename,final_optimal_featureset.basis_vector_matrix[i].data_matrix,sorted_number_of_features[i],number_of_new_features);
	}
	strcpy(filename,path);
	strcat(filename,"canonical_variables.txt");
	write_output_file(filename,final_optimal_featureset.canonical_variables_matrix.data_matrix,class_labels,number_of_samples,number_of_new_features,number_of_class_labels);
	//strcpy(filename,path);
	//strcat(filename,"correlation.txt");
	//write_correlation_file(filename,final_optimal_featureset.correlation,number_of_new_features);
	strcpy(filename,path);
	strcat(filename,"feature_from_modality_index.txt");
	write_index_file(filename,feature_from_modalities,number_of_new_features);
	free(index);
	for(i=0;i<number_of_modalities;i++)
	{
		for(j=0;j<number_of_modalities;j++)
			double_matrix_deallocation(sorted_covariance_matrix[i][j].data_matrix,sorted_number_of_features[i]);
		free(sorted_covariance_matrix[i]);
	}
	free(sorted_covariance_matrix);
	for(i=0;i<number_of_modalities;i++)
		double_matrix_deallocation(sorted_inverse_covariance_matrix[i].data_matrix,sorted_number_of_features[i]);
	free(sorted_inverse_covariance_matrix);
	free(combination_number);
	free(feature_from_modalities);
	int_matrix_deallocation(temp_data_matrix,count*count);
	for(i=0;i<number_of_modalities-1;i++)
	{
		for(j=0;j<number_of_new_features;j++)
			int_matrix_deallocation(optimal_featureset[i].rhepm_data_matrix[j].data_matrix,number_of_class_labels);
		free(optimal_featureset[i].rhepm_data_matrix);
		for(j=0;j<number_of_modalities;j++)
			double_matrix_deallocation(optimal_featureset[i].basis_vector_matrix[j].data_matrix,sorted_number_of_features[j]);
		free(optimal_featureset[i].basis_vector_matrix);
		double_matrix_deallocation(optimal_featureset[i].canonical_variables_matrix.data_matrix,number_of_samples);
		free(optimal_featureset[i].correlation);
		free(optimal_featureset[i].relevance);
		free(optimal_featureset[i].objective_function_value);
	}
	free(optimal_featureset);
	for(i=0;i<number_of_new_features;i++)
		int_matrix_deallocation(final_optimal_featureset.rhepm_data_matrix[i].data_matrix,number_of_class_labels);
	free(final_optimal_featureset.rhepm_data_matrix);
	for(i=0;i<number_of_modalities;i++)
		double_matrix_deallocation(final_optimal_featureset.basis_vector_matrix[i].data_matrix,sorted_number_of_features[i]);
	free(final_optimal_featureset.basis_vector_matrix);
	double_matrix_deallocation(final_optimal_featureset.canonical_variables_matrix.data_matrix,number_of_samples);
	free(final_optimal_featureset.correlation);
	free(final_optimal_featureset.relevance);
	free(final_optimal_featureset.objective_function_value);
	free(new_combination_number);
	for(i=0;i<number_of_modalities;i++)
		double_matrix_deallocation(basis_vector[i].data_matrix,sorted_number_of_features[i]);
	free(basis_vector);
	free(sorted_number_of_features);
	free(filename);
	free(temp);
	delete_file(path,number_of_modalities,count);
}

void between_set_covariance(double **data_matrix1,double **data_matrix2,double **new_data_matrix,int row1,int row2,int column)
{
	int i,j;
	double **transpose_data_matrix;

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
	int i,j;
	double **transpose_data_matrix;

	transpose_data_matrix=double_matrix_allocation(column,row);

	matrix_transpose(data_matrix,transpose_data_matrix,column,row);
	matrix_multiplication(data_matrix,transpose_data_matrix,new_data_matrix,row,column,row);
	for(i=0;i<row;i++)
		for(j=0;j<row;j++)
			new_data_matrix[i][j]=new_data_matrix[i][j]/column;

	double_matrix_deallocation(transpose_data_matrix,column);
}

void empirical_unbiased_between_set_variance(double **data_matrix1,double **data_matrix2,double **covariance_data_matrix,double **new_data_matrix,int row1,int row2,int column)
{
	int i,j,k;
	double sum1,sum2;
	double optimal_lambda;
	double **transpose_data_matrix,**unbiased_variance_data_matrix,**variance_of_unbiased_variance_data_matrix;
	double ***data_vector;

	sum1=0;
	sum2=0;
	optimal_lambda=0;
	transpose_data_matrix=double_matrix_allocation(column,row2);
	unbiased_variance_data_matrix=double_matrix_allocation(row1,row2);
	variance_of_unbiased_variance_data_matrix=double_matrix_allocation(row1,row2);
	data_vector=(double ***)malloc(sizeof(double **)*column);
	for(i=0;i<column;i++)
		data_vector[i]=double_matrix_allocation(row1,row2);
	matrix_transpose(data_matrix2,transpose_data_matrix,column,row2);
	for(i=0;i<row1;i++)
		for(j=0;j<row2;j++)
			for(k=0;k<column;k++)
				data_vector[k][i][j]=data_matrix1[i][k]*transpose_data_matrix[k][j];
	for(i=0;i<row1;i++)
		for(j=0;j<row2;j++)
			unbiased_variance_data_matrix[i][j]=(covariance_data_matrix[i][j]*column)/(column-1);
	for(i=0;i<row1;i++)
		for(j=0;j<row2;j++)
		{
			variance_of_unbiased_variance_data_matrix[i][j]=0;
			for(k=0;k<column;k++)
				variance_of_unbiased_variance_data_matrix[i][j]+=pow((data_vector[k][i][j]-covariance_data_matrix[i][j]),2);
			variance_of_unbiased_variance_data_matrix[i][j]=(variance_of_unbiased_variance_data_matrix[i][j]*column)/((column-1)*(column-1)*(column-1));
		}
	for(i=0;i<row1;i++)
		for(j=0;j<row2;j++)
			if(i!=j)
			{
				sum1+=variance_of_unbiased_variance_data_matrix[i][j];
				sum2+=pow(unbiased_variance_data_matrix[i][j],2);
			}
	optimal_lambda=sum1/sum2;
	for(i=0;i<row1;i++)
		for(j=0;j<row2;j++)
			new_data_matrix[i][j]=(1-optimal_lambda)*unbiased_variance_data_matrix[i][j];
	double_matrix_deallocation(transpose_data_matrix,column);
	double_matrix_deallocation(unbiased_variance_data_matrix,row1);
	double_matrix_deallocation(variance_of_unbiased_variance_data_matrix,row1);
	for(i=0;i<column;i++)
		double_matrix_deallocation(data_vector[i],row1);
	free(data_vector);
}

void empirical_unbiased_within_set_variance(double **data_matrix,double **covariance_data_matrix,double **new_data_matrix,int row,int column)
{
	int i,j,k;
	double sum1, sum2;
	double optimal_lambda;
	double **transpose_data_matrix,**unbiased_variance_data_matrix,**variance_of_unbiased_variance_data_matrix;
	double ***data_vector;

	sum1=0;
	sum2=0;
	optimal_lambda=0;
	transpose_data_matrix=double_matrix_allocation(column,row);
	unbiased_variance_data_matrix=double_matrix_allocation(row,row);
	variance_of_unbiased_variance_data_matrix=double_matrix_allocation(row,row);
	data_vector=(double ***)malloc(sizeof(double **)*column);
	for(i=0;i<column;i++)
		data_vector[i]=double_matrix_allocation(row,row);
	matrix_transpose(data_matrix,transpose_data_matrix,column,row);
	for(i=0;i<row;i++)
		for(j=0;j<row;j++)
			for(k=0;k<column;k++)
				data_vector[k][i][j]=data_matrix[i][k]*transpose_data_matrix[k][j];
	for(i=0;i<row;i++)
		for(j=0;j<row;j++)
			unbiased_variance_data_matrix[i][j]=(covariance_data_matrix[i][j]*column)/(column-1);
	for(i=0;i<row;i++)
		for(j=0;j<row;j++)
		{
			variance_of_unbiased_variance_data_matrix[i][j]=0;
			for(k=0;k<column;k++)
				variance_of_unbiased_variance_data_matrix[i][j]+=pow((data_vector[k][i][j]-covariance_data_matrix[i][j]),2);
			variance_of_unbiased_variance_data_matrix[i][j]=(variance_of_unbiased_variance_data_matrix[i][j]*column)/((column-1)*(column-1)*(column-1));
		}
	for(i=0;i<row;i++)
		for(j=0;j<row;j++)
			if(i!=j)
			{
				sum1+=variance_of_unbiased_variance_data_matrix[i][j];
				sum2+=pow(unbiased_variance_data_matrix[i][j],2);
			}
	optimal_lambda=sum1/sum2;
	for(i=0;i<row;i++)
		for(j=0;j<row;j++)
		{
			if(i==j)
				new_data_matrix[i][j]=unbiased_variance_data_matrix[i][j];
			else
				new_data_matrix[i][j]=(1 - optimal_lambda)*unbiased_variance_data_matrix[i][j];
		}
	double_matrix_deallocation(transpose_data_matrix,column);
	double_matrix_deallocation(unbiased_variance_data_matrix,row);
	double_matrix_deallocation(variance_of_unbiased_variance_data_matrix,row);
	for(i=0;i<column;i++)
		double_matrix_deallocation(data_vector[i],row);
	free(data_vector);
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
	read_eigenvector_file(eigenvector_filename,eigenvector_data_matrix,size_of_matrix);
	read_eigenvalue_file(eigenvalue_filename,eigenvalue_data_matrix,size_of_matrix);

	foldername=(char *)malloc(sizeof(char)*1000);
	strcpy(foldername,"rm -f ");
	strcat(foldername,data_filename);
	strcat(foldername," ");
	strcat(foldername,eigenvector_filename);
	strcat(foldername," ");
	strcat(foldername,eigenvalue_filename);
	strcat(foldername," eigenvalue_eigenvector.R eigenvalue_eigenvector.Rout");
	system(foldername);

	free(data_filename);
	free(eigenvector_filename);
	free(eigenvalue_filename);
	free(foldername);
}

int rhepm(struct featureset *optimal_featureset,struct featureset final_optimal_featureset,double **canonical_variables_data_matrix,int *class_labels,int *combination_number,int *feature_from_modalities,int number_of_modalities,int number_of_samples,int number_of_new_features,int combination,int number_of_class_labels,double epsilon,double omega)
{
	int i,j;
	int number;
	int flag;
	double significance;
	double joint_dependency;
	double maximum_objective_function_value;
	double *each_feature_data_matrix;
	int **resultant_equivalence_partition;
	struct feature *new_feature;

	flag=0;
	new_feature=(struct feature *)malloc(sizeof(struct feature)*combination);
	for(i=0;i<combination;i++)
	{
		each_feature_data_matrix=(double *)malloc(sizeof(double)*number_of_samples);
		new_feature[i].rhepm_data_matrix=int_matrix_allocation(number_of_class_labels,number_of_samples);
		for(j=0;j<number_of_samples;j++)
			each_feature_data_matrix[j]=canonical_variables_data_matrix[j][i];
		generate_equivalence_partition(new_feature[i].rhepm_data_matrix,each_feature_data_matrix,class_labels,number_of_samples,number_of_class_labels,epsilon,i);
		new_feature[i].relevance=dependency_degree(new_feature[i].rhepm_data_matrix,number_of_samples,number_of_class_labels);
		free(each_feature_data_matrix);
	}
	for(i=0;i<combination;i++)
	{
		new_feature[i].significance=0;
		for(j=0;j<number_of_new_features;j++)
		{
			resultant_equivalence_partition=int_matrix_allocation(number_of_class_labels,number_of_samples);
			form_resultant_equivalence_partition_matrix(new_feature[i].rhepm_data_matrix,final_optimal_featureset.rhepm_data_matrix[j].data_matrix,resultant_equivalence_partition,number_of_class_labels,number_of_samples);
			joint_dependency=dependency_degree(resultant_equivalence_partition,number_of_samples,number_of_class_labels);
			new_feature[i].significance+=joint_dependency-final_optimal_featureset.relevance[j];
			int_matrix_deallocation(resultant_equivalence_partition,number_of_class_labels);
		}
		if(number_of_new_features)
			new_feature[i].significance/=number_of_new_features;
		new_feature[i].objective_function_value=omega*new_feature[i].relevance+(1-omega)*new_feature[i].significance;
	}
	number=0;
	maximum_objective_function_value=new_feature[0].objective_function_value;
	for(i=1;i<combination;i++)
		if(maximum_objective_function_value<new_feature[i].objective_function_value)
		{
			number=i;
			maximum_objective_function_value=new_feature[i].objective_function_value;
		}
	optimal_featureset[number_of_modalities-2].relevance[number_of_new_features]=new_feature[number].relevance;
	final_optimal_featureset.relevance[number_of_new_features]=optimal_featureset[number_of_modalities-2].relevance[number_of_new_features];
	optimal_featureset[number_of_modalities-2].objective_function_value[number_of_new_features]=new_feature[number].objective_function_value;
	final_optimal_featureset.objective_function_value[number_of_new_features]=optimal_featureset[number_of_modalities-2].objective_function_value[number_of_new_features];
	for(i=0;i<number_of_class_labels;i++)
		for(j=0;j<number_of_samples;j++)
		{
			optimal_featureset[number_of_modalities-2].rhepm_data_matrix[number_of_new_features].data_matrix[i][j]=new_feature[number].rhepm_data_matrix[i][j];
			final_optimal_featureset.rhepm_data_matrix[number_of_new_features].data_matrix[i][j]=optimal_featureset[number_of_modalities-2].rhepm_data_matrix[number_of_new_features].data_matrix[i][j];
		}
	if(number_of_modalities>2)
	{
		if(!number_of_new_features)
		{
			if(optimal_featureset[number_of_modalities-3].relevance[number_of_new_features]>final_optimal_featureset.relevance[number_of_new_features])
			{
				flag=1;
				final_optimal_featureset.relevance[number_of_new_features]=optimal_featureset[number_of_modalities-3].relevance[number_of_new_features];
				final_optimal_featureset.objective_function_value[number_of_new_features]=optimal_featureset[number_of_modalities-3].objective_function_value[number_of_new_features];
				for(i=0;i<number_of_class_labels;i++)
					for(j=0;j<number_of_samples;j++)
						final_optimal_featureset.rhepm_data_matrix[number_of_new_features].data_matrix[i][j]=optimal_featureset[number_of_modalities-3].rhepm_data_matrix[number_of_new_features].data_matrix[i][j];
			}
			else
				feature_from_modalities[number_of_new_features]=number_of_modalities;
		}
		else
		{
			if(optimal_featureset[number_of_modalities-3].objective_function_value[number_of_new_features]>final_optimal_featureset.objective_function_value[number_of_new_features])
			{
				flag=1;
				final_optimal_featureset.relevance[number_of_new_features]=optimal_featureset[number_of_modalities-3].relevance[number_of_new_features];
				final_optimal_featureset.objective_function_value[number_of_new_features]=optimal_featureset[number_of_modalities-3].objective_function_value[number_of_new_features];
				for(i=0;i<number_of_class_labels;i++)
					for(j=0;j<number_of_samples;j++)
						final_optimal_featureset.rhepm_data_matrix[number_of_new_features].data_matrix[i][j]=optimal_featureset[number_of_modalities-3].rhepm_data_matrix[number_of_new_features].data_matrix[i][j];
			}
			else
				feature_from_modalities[number_of_new_features]=number_of_modalities;
		}
	}
	for(i=0;i<combination;i++)
		int_matrix_deallocation(new_feature[i].rhepm_data_matrix,number_of_class_labels);
	free(new_feature);
	combination_number[number_of_new_features]=number;

	return flag;
}

void generate_equivalence_partition(int **rhepm_data_matrix,double *data_matrix,int *class_labels,int number_of_samples,int number_of_class_labels,double epsilon,int combination)
{
	int i,j,k,l;
	double minimum,maximum;
	int *label;
	double *minimum_data_matrix,*maximum_data_matrix;

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
	int sum;
	int i,j;
	double gamma;
	int *confusion_vector;

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
			fprintf(fp_write,"%.20lf\t",new_data_matrix[i][j]);
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
			fprintf(fp_write,"%e\t",new_data_matrix[i][j]);
		fprintf(fp_write,"\n");
	}
	fclose(fp_write);
}

void write_index_file(char *filename,int *new_data_matrix,int size_of_matrix)
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
		fprintf(fp_write,"%d\n",new_data_matrix[i]);
	fclose(fp_write);
}

void write_20_decimal_paces_file(char *filename,double **new_data_matrix,int row,int column)
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
			fprintf(fp_write,"%.20lf\t",new_data_matrix[i][j]);
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
		fprintf(fp_write,"%.20lf\n",new_data_matrix[i]);
	fclose(fp_write);
}

void write_lambda_file(char *filename,double **new_data_matrix,int row,int column)
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

void delete_file(char *path,int number_of_modalities,int count)
{
	int i,j,t;
	char *filename,*temp;

	filename=(char *)malloc(sizeof(char)*1000);
	temp=(char *)malloc(sizeof(char)*1000);
	for(t=0;t<count;t++)
	{
		for(i=0;i<number_of_modalities;i++)
		{
			strcpy(filename,path);
			sprintf(temp,"inverse_covariance_data_matrix_%d_%d.txt",i+1,t+1);
			strcat(filename,temp);
			strcpy(temp,"rm -f ");
			strcat(temp,filename);
			system(temp);
			strcpy(filename,path);
			sprintf(temp,"eigenvector%d_%d.txt",i+1,t+1);
			strcat(filename,temp);
			strcpy(temp,"rm -f ");
			strcat(temp,filename);
			system(temp);
		}
	}
	for(i=0;i<count;i++)
	{
		for(j=0;j<count;j++)
		{
			strcpy(filename,path);
			sprintf(temp,"eigenvector1_%d_%d.txt",i+1,j+1);
			strcat(filename,temp);
			strcpy(temp,"rm -f ");
			strcat(temp,filename);
			system(temp);
			strcpy(filename,path);
			sprintf(temp,"eigenvector2_%d_%d.txt",i+1,j+1);
			strcat(filename,temp);
			strcpy(temp,"rm -f ");
			strcat(temp,filename);
			system(temp);
			strcpy(filename,path);
			sprintf(temp,"correlation_%d_%d.txt",i+1,j+1);
			strcat(filename,temp);
			strcpy(temp,"rm -f ");
			strcat(temp,filename);
			system(temp);
			strcpy(filename,path);
			sprintf(temp,"canonical_variables_%d_%d.txt",i+1,j+1);
			strcat(filename,temp);
			strcpy(temp,"rm -f ");
			strcat(temp,filename);
			system(temp);
		}
		strcpy(filename,path);
		sprintf(temp,"correlation_%d.txt",i+1);
		strcat(filename,temp);
		strcpy(temp,"rm -f ");
		strcat(temp,filename);
		system(temp);
		strcpy(filename,path);
		sprintf(temp,"canonical_variables_%d.txt",i+1);
		strcat(filename,temp);
		strcpy(temp,"rm -f ");
		strcat(temp,filename);
		system(temp);
	}
	free(filename);
	free(temp);
}

void char_matrix_deallocation(char **data,int row)
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

void double_matrix_deallocation(double **data,int row)
{
	int i;

	for(i=0;i<row;i++)
		free(data[i]);
	free(data);
}
