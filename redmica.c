/*********************************************************************/
/*Source Code of ReDMiCA Algorithm written in C and R                */
/*A. Mandal and P. Maji, ReDMiCA: Multiview Regularized Discriminant */
/*Canonical Correlation Analysis: Sequential Extraction of Relevant  */
/*Features from Multiblock Data,                                     */
/*IEEE Transactions on Cybernetics, 2022.                            */
/*********************************************************************/

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
	double *relevance;
	double *correlation;
	struct rhepm_matrix *rhepm_data_matrix;
	struct modality canonical_variables_matrix;
	struct modality *basis_vector_matrix;
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
void preprocessing(char *,char *,int,int,double,double,double,double,double,int,int);
void read_filename(char **,char *,int);
struct dataset read_input_file(char *);
void read_input_canonical_file(char *,double **,int *);
void read_file(char *,double **,int,int);
void read_rhepm_file(char *,int **,int,int);
void read_eigenvector_file(char *,double **,int);
void read_eigenvalue_file(char *,double *,int);
void zero_mean(double **,double **,int,int);
void ReDMiCA(char *,struct modality *,struct modality *,int *,int *,int,int,int,int,double,double,double,double,double,int,int);
void between_set_covariance(double **,double **,double **,int,int,int);
void within_set_covariance(double **,double **,int,int);
void matrix_transpose(double **,double **,int,int);
void matrix_multiplication(double **,double **,double **,int,int,int);
void eigenvalue_eigenvector(char *,double **,double **,double *,int);
void copy_eigenvector(double **,double **,int,int);
int rhepm(struct featureset,double **,double **,int *,int,int,int,int,int,double,double);
void generate_equivalence_partition(int **,double *,int *,int,int,double);
double dependency_degree(int **,int,int);
double form_resultant_equivalence_partition_matrix(int **,int **,int **,int,int);
void write_output_file(char *,double **,int *,int,int,int);
void write_file(char *,double **,int,int);
void write_20_decimal_paces_file(char *,double **,int,int);
void write_rhepm_file(char *,int **,int,int);
void write_correlation_file(char *,double *,int);
void write_relevance_file(char *,double *,int);
void write_lambda_file(char *,double **,int,int);
void write_R_eigenvalue_eigenvector(char *,char *,char *,char *);
void char_matrix_deallocation(char **,int);
void int_matrix_deallocation(int **,int);
void double_matrix_deallocation(double **,int);

int main(int argc,char *argv[])
{
	int o;
	int number_of_modalities,number_of_new_features;
	int starting_combination;
	int extracted_feature_number;
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
	starting_combination=0;
	extracted_feature_number=0;

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
	preprocessing(modality_filename,path,number_of_modalities,number_of_new_features,lambda_minimum,lambda_maximum,delta,epsilon,omega,starting_combination,extracted_feature_number);
	(void)ftime(&tf);
	printf("\nTOTAL TIME REQUIRED for ReDMiCA=%d millisec\n",(int)(1000.0*(tf.time-ti.time)+(tf.millitm-ti.millitm)));
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

void preprocessing(char *modality_filename,char *path,int number_of_modalities,int number_of_new_features,double lambda_minimum,double lambda_maximum,double delta,double epsilon,double omega,int starting_combination,int extracted_feature_number)
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
	strcat(filename,"ReDMiCA/");
	if(stat(filename,&st)==-1)
	    	mkdir(filename,0700);
	ReDMiCA(filename,transpose_matrix,zero_mean_matrix,Dataset[0].class_labels,number_of_features,Dataset[0].number_of_samples,number_of_new_features,Dataset[0].number_of_class_labels,number_of_modalities,lambda_minimum,lambda_maximum,delta,epsilon,omega,starting_combination,extracted_feature_number);

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

void read_input_canonical_file(char *filename,double **data_matrix,int *class_labels)
{
	int i,j;
	int number_of_samples,number_of_features,number_of_class_labels;
	FILE *fp_read;

	fp_read=fopen(filename,"r");
	if(fp_read==NULL)
	{
		printf("\nError: Error in Input File.\n");
		exit(0);
	}
	fscanf(fp_read,"%d%d%d",&number_of_samples,&number_of_features,&number_of_class_labels); 
	for(i=0;i<number_of_samples;i++)
	{
		for(j=0;j<number_of_features;j++)
			fscanf(fp_read,"%lf",&data_matrix[i][j]);
		fscanf(fp_read,"%d",&class_labels[i]);
	}
	fclose(fp_read);
}

void read_file(char *filename,double **data_matrix,int row,int column)
{
	int i,j;
	FILE *fp_read;

	fp_read=fopen(filename,"r");
	if(fp_read==NULL)
	{
		printf("\nError: Error in Input File.\n");
		exit(0);
	}
	for(i=0;i<row;i++)
		for(j=0;j<column;j++)
			fscanf(fp_read,"%lf",&data_matrix[i][j]);
	fclose(fp_read);
}

void read_rhepm_file(char *filename,int **data_matrix,int row,int column)
{
	int i,j;
	FILE *fp_read;

	fp_read=fopen(filename,"r");
	if(fp_read==NULL)
	{
		printf("\nError: Error in Input File.\n");
		exit(0);
	}
	for(i=0;i<row;i++)
		for(j=0;j<column;j++)
			fscanf(fp_read,"%d",&data_matrix[i][j]);
	fclose(fp_read);
}

void read_eigenvector_file(char *filename,double **data_matrix,int size_of_matrix)
{
	int i,j;
	FILE *fp_read; 

	fp_read=fopen(filename,"r");
	if(filename==NULL)
	{
		printf("\nError: Error in Input File.\n");
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
		printf("\nError: Error in Input File.\n");
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

void ReDMiCA(char *path,struct modality *transpose_matrix,struct modality *zero_mean_matrix,int *class_labels,int *number_of_features,int number_of_samples,int number_of_new_features,int number_of_class_labels,int number_of_modalities,double lambda_minimum,double lambda_maximum,double delta,double epsilon,double omega,int starting_combination,int extracted_feature_number)
{
	int i,j,k,l,m,n,t;
	int count,combination;
	int combination_number;
	int flag;
	int number_of_merged_features;
	char *filename,*temp;
	int *regularization;
	double *eigenvalue;
	double *correlation;
	int **regularization_pointer;
	double **eigenvalue_of_covariance_data_matrix;
	double **temp_data_matrix1,**temp_data_matrix2,**temp_data_matrix3;
	double **h_data_matrix;
	double **eigenvector;
	double **basis_vector_data_matrix,**transpose_basis_vector_data_matrix;
	double **canonical_variables_data_matrix;
	double **lambda;
	double **optimal_lambda_objective_function_value;
	double **new_data_matrix;
	double **zero_mean_data_matrix;
	double **covariance_data_matrix;
	double **eigenvector_data_matrix;
	struct featureset new_featureset,optimal_featureset;
	struct modality *eigenvector_of_covariance_data_matrix,*transpose_eigenvector_of_covariance_data_matrix;
	struct modality *inverse_covariance_matrix;
	struct modality *canonical_variables;
	struct modality **covariance_matrix;
	struct modality **multiplication_inverse_covariance_matrix1,**multiplication_inverse_covariance_matrix2;

	count=(int)((lambda_maximum-lambda_minimum)/delta)+1;
	combination=pow(count,number_of_modalities);
	filename=(char *)malloc(sizeof(char)*1000);
	temp=(char *)malloc(sizeof(char)*1000);
	regularization=(int *)malloc(sizeof(int)*number_of_modalities);
	regularization_pointer=int_matrix_allocation(combination,number_of_modalities);
	lambda=double_matrix_allocation(combination,number_of_modalities);
	optimal_lambda_objective_function_value=double_matrix_allocation(number_of_new_features,number_of_modalities+1);
	new_featureset.correlation=(double *)malloc(sizeof(double)*number_of_new_features);
	optimal_featureset.correlation=(double *)malloc(sizeof(double)*number_of_new_features);
	new_featureset.canonical_variables_matrix.data_matrix=double_matrix_allocation(number_of_samples,number_of_new_features);
	optimal_featureset.canonical_variables_matrix.data_matrix=double_matrix_allocation(number_of_samples,number_of_new_features);
	new_featureset.basis_vector_matrix=(struct modality *)malloc(sizeof(struct modality)*number_of_modalities);
	optimal_featureset.basis_vector_matrix=(struct modality *)malloc(sizeof(struct modality)*number_of_modalities);
	for(i=0;i<number_of_modalities;i++)
	{
		new_featureset.basis_vector_matrix[i].data_matrix=double_matrix_allocation(number_of_features[i],number_of_new_features);
		optimal_featureset.basis_vector_matrix[i].data_matrix=double_matrix_allocation(number_of_features[i],number_of_new_features);
	}
	optimal_featureset.relevance=(double *)malloc(sizeof(double)*number_of_new_features);
	optimal_featureset.rhepm_data_matrix=(struct rhepm_matrix *)malloc(sizeof(struct rhepm_matrix)*number_of_new_features);
	for(i=0;i<number_of_new_features;i++)
		optimal_featureset.rhepm_data_matrix[i].data_matrix=int_matrix_allocation(number_of_class_labels,number_of_samples);
	for(i=0;i<number_of_modalities;i++)
		regularization[i]=0;
	for(i=0;i<combination;i++)
	{
		for(j=0;j<number_of_modalities;j++)
		{
			regularization_pointer[i][j]=regularization[j];
			if(j==number_of_modalities-1)
			{
				if(regularization[j]==count-1)
				{
					regularization[j]=0;
					for(k=j-1;k>=0;k--)
					{
						if(regularization[k]==count-1)
							regularization[k]=0;
						else
						{
							regularization[k]++;
							break;
						}
					}
				}
				else
					regularization[j]++;
			}
		}
	}
	free(regularization);
	eigenvalue_of_covariance_data_matrix=(double **)malloc(sizeof(double *)*number_of_modalities);
	eigenvector_of_covariance_data_matrix=(struct modality *)malloc(sizeof(struct modality)*number_of_modalities);
	transpose_eigenvector_of_covariance_data_matrix=(struct modality *)malloc(sizeof(struct modality)*number_of_modalities);
	inverse_covariance_matrix=(struct modality *)malloc(sizeof(struct modality)*count);
	covariance_matrix=(struct modality **)malloc(sizeof(struct modality *)*number_of_modalities);
	for(i=0;i<number_of_modalities;i++)
	{
		covariance_matrix[i]=(struct modality *)malloc(sizeof(struct modality)*number_of_modalities);
		for(j=0;j<number_of_modalities;j++)
			covariance_matrix[i][j].data_matrix=double_matrix_allocation(number_of_features[i],number_of_features[j]);
		eigenvalue_of_covariance_data_matrix[i]=(double *)malloc(sizeof(double)*number_of_features[i]);
		eigenvector_of_covariance_data_matrix[i].data_matrix=double_matrix_allocation(number_of_features[i],number_of_features[i]);
		transpose_eigenvector_of_covariance_data_matrix[i].data_matrix=double_matrix_allocation(number_of_features[i],number_of_features[i]);
	}
	multiplication_inverse_covariance_matrix1=(struct modality **)malloc(sizeof(struct modality *)*count);
	multiplication_inverse_covariance_matrix2=(struct modality **)malloc(sizeof(struct modality *)*count);
	for(i=0;i<count;i++)
	{
		inverse_covariance_matrix[i].data_matrix=double_matrix_allocation(number_of_features[0],number_of_features[0]);
		multiplication_inverse_covariance_matrix1[i]=(struct modality *)malloc(sizeof(struct modality)*(number_of_modalities-1));
		multiplication_inverse_covariance_matrix2[i]=(struct modality *)malloc(sizeof(struct modality)*(number_of_modalities-1));
		for(j=0;j<number_of_modalities-1;j++)
		{
			multiplication_inverse_covariance_matrix1[i][j].data_matrix=double_matrix_allocation(number_of_features[0],number_of_features[0]);
			multiplication_inverse_covariance_matrix2[i][j].data_matrix=double_matrix_allocation(number_of_features[j+1],number_of_features[j]);
		}
	}
	if(!starting_combination)
	{
		for(i=0;i<number_of_modalities;i++)
		{
			for(j=0;j<number_of_modalities;j++)
			{
				if(i==j)
				{
					within_set_covariance(zero_mean_matrix[i].data_matrix,covariance_matrix[i][i].data_matrix,number_of_features[i],number_of_samples);
					if(lambda_minimum)
					{
						for(k=0;k<number_of_features[i];k++)
							covariance_matrix[i][i].data_matrix[k][k]+=lambda_minimum;
					}
				}
				else if(i<j)
					between_set_covariance(zero_mean_matrix[i].data_matrix,zero_mean_matrix[j].data_matrix,covariance_matrix[i][j].data_matrix,number_of_features[i],number_of_features[j],number_of_samples);
				else
					matrix_transpose(covariance_matrix[j][i].data_matrix,covariance_matrix[i][j].data_matrix,number_of_features[i],number_of_features[j]);
			}
			eigenvalue_eigenvector(path,covariance_matrix[i][i].data_matrix,eigenvector_of_covariance_data_matrix[i].data_matrix,eigenvalue_of_covariance_data_matrix[i],number_of_features[i]);
			matrix_transpose(eigenvector_of_covariance_data_matrix[i].data_matrix,transpose_eigenvector_of_covariance_data_matrix[i].data_matrix,number_of_features[i],number_of_features[i]);
		}
		for(t=0;t<count;t++)
			for(i=0;i<number_of_modalities;i++)
			{
				temp_data_matrix1=double_matrix_allocation(number_of_features[i],number_of_features[i]);
				for(j=0;j<number_of_features[i];j++)
					for(k=0;k<number_of_features[i];k++)
					{
						if((1/(eigenvalue_of_covariance_data_matrix[i][k]+t*delta))>=0.000001)
							temp_data_matrix1[j][k]=eigenvector_of_covariance_data_matrix[i].data_matrix[j][k]/(eigenvalue_of_covariance_data_matrix[i][k]+t*delta);
						else
							temp_data_matrix1[j][k]=eigenvector_of_covariance_data_matrix[i].data_matrix[j][k]/0.000001;
					}
				if(i)
				{
					temp_data_matrix2=double_matrix_allocation(number_of_features[i],number_of_features[i]);
					temp_data_matrix3=double_matrix_allocation(number_of_features[0],number_of_features[i]);
					matrix_multiplication(temp_data_matrix1,transpose_eigenvector_of_covariance_data_matrix[i].data_matrix,temp_data_matrix2,number_of_features[i],number_of_features[i],number_of_features[i]);
					matrix_multiplication(covariance_matrix[0][i].data_matrix,temp_data_matrix2,temp_data_matrix3,number_of_features[0],number_of_features[i],number_of_features[i]);
					matrix_multiplication(temp_data_matrix3,covariance_matrix[i][0].data_matrix,multiplication_inverse_covariance_matrix1[t][i-1].data_matrix,number_of_features[0],number_of_features[i],number_of_features[0]);
					matrix_multiplication(temp_data_matrix2,covariance_matrix[i][i-1].data_matrix,multiplication_inverse_covariance_matrix2[t][i-1].data_matrix,number_of_features[i],number_of_features[i],number_of_features[i-1]);
					strcpy(filename,path);
					sprintf(temp,"multiplication_inverse_covariance_matrix1_%d_%d.txt",t,i-1);
					strcat(filename,temp);
					write_20_decimal_paces_file(filename,multiplication_inverse_covariance_matrix1[t][i-1].data_matrix,number_of_features[0],number_of_features[0]);
					strcpy(filename,path);
					sprintf(temp,"multiplication_inverse_covariance_matrix2_%d_%d.txt",t,i-1);
					strcat(filename,temp);
					write_20_decimal_paces_file(filename,multiplication_inverse_covariance_matrix2[t][i-1].data_matrix,number_of_features[i],number_of_features[i-1]);
					double_matrix_deallocation(temp_data_matrix2,number_of_features[i]);
					double_matrix_deallocation(temp_data_matrix3,number_of_features[0]);
				}
				else
				{
					matrix_multiplication(temp_data_matrix1,transpose_eigenvector_of_covariance_data_matrix[i].data_matrix,inverse_covariance_matrix[t].data_matrix,number_of_features[0],number_of_features[0],number_of_features[0]);
					strcpy(filename,path);
					sprintf(temp,"inverse_covariance_matrix_%d.txt",t);
					strcat(filename,temp);
					write_20_decimal_paces_file(filename,inverse_covariance_matrix[t].data_matrix,number_of_features[0],number_of_features[0]);
				}
				double_matrix_deallocation(temp_data_matrix1,number_of_features[i]);
			}
	}
	else
	{
		for(t=0;t<count;t++)
			for(i=0;i<number_of_modalities;i++)
			{
				if(i)
				{
					strcpy(filename,path);
					sprintf(temp,"multiplication_inverse_covariance_matrix1_%d_%d.txt",t,i-1);
					strcat(filename,temp);
					read_file(filename,multiplication_inverse_covariance_matrix1[t][i-1].data_matrix,number_of_features[0],number_of_features[0]);
					strcpy(filename,path);
					sprintf(temp,"multiplication_inverse_covariance_matrix2_%d_%d.txt",t,i-1);
					strcat(filename,temp);
					read_file(filename,multiplication_inverse_covariance_matrix2[t][i-1].data_matrix,number_of_features[i],number_of_features[i-1]);
				}
				else
				{
					strcpy(filename,path);
					sprintf(temp,"inverse_covariance_matrix_%d.txt",t);
					strcat(filename,temp);
					read_file(filename,inverse_covariance_matrix[t].data_matrix,number_of_features[0],number_of_features[0]);
				}
			}
	}
	for(i=0;i<number_of_modalities;i++)
	{
		free(eigenvalue_of_covariance_data_matrix[i]);
		double_matrix_deallocation(eigenvector_of_covariance_data_matrix[i].data_matrix,number_of_features[i]);
		double_matrix_deallocation(transpose_eigenvector_of_covariance_data_matrix[i].data_matrix,number_of_features[i]);
	}
	free(eigenvalue_of_covariance_data_matrix);
	free(eigenvector_of_covariance_data_matrix);
	free(transpose_eigenvector_of_covariance_data_matrix);
	for(i=0;i<number_of_modalities;i++)
	{
		for(j=0;j<number_of_modalities;j++)
			double_matrix_deallocation(covariance_matrix[i][j].data_matrix,number_of_features[i]);
		free(covariance_matrix[i]);
	}
	free(covariance_matrix);
	h_data_matrix=double_matrix_allocation(number_of_features[0],number_of_features[0]);
	canonical_variables_data_matrix=double_matrix_allocation(number_of_new_features,number_of_samples);
	temp_data_matrix1=double_matrix_allocation(number_of_features[0],number_of_features[0]);
	temp_data_matrix2=double_matrix_allocation(number_of_features[0],number_of_features[0]);
	for(i=0;i<number_of_features[0];i++)
		for(j=0;j<number_of_features[0];j++)
			h_data_matrix[i][j]=0;
	if(starting_combination)
	{
		strcpy(filename,path);
		sprintf(temp,"temp_lambda_%d.txt",starting_combination);
		strcat(filename,temp);
		read_file(filename,lambda,starting_combination,number_of_modalities);
	}
	for(t=starting_combination;t<combination;t++)
	{
		for(i=0;i<number_of_samples;i++)
			for(j=0;j<number_of_new_features;j++)
				new_featureset.canonical_variables_matrix.data_matrix[i][j]=0;
		if(!(t%count))
		{
			for(i=0;i<number_of_features[0];i++)
				for(j=0;j<number_of_features[0];j++)
					h_data_matrix[i][j]=0;
			for(i=1;i<number_of_modalities;i++)
			{
				if(i<number_of_modalities-1)
				{
					matrix_multiplication(inverse_covariance_matrix[regularization_pointer[t][0]].data_matrix,multiplication_inverse_covariance_matrix1[regularization_pointer[t][i]][i-1].data_matrix,temp_data_matrix1,number_of_features[0],number_of_features[0],number_of_features[0]);
					for(j=0;j<number_of_features[0];j++)
						for(k=0;k<number_of_features[0];k++)
							h_data_matrix[j][k]+=temp_data_matrix1[j][k];
				}
			}
		}
		matrix_multiplication(inverse_covariance_matrix[regularization_pointer[t][0]].data_matrix,multiplication_inverse_covariance_matrix1[regularization_pointer[t][number_of_modalities-1]][number_of_modalities-2].data_matrix,temp_data_matrix1,number_of_features[0],number_of_features[0],number_of_features[0]);
		for(i=0;i<number_of_features[0];i++)
			for(j=0;j<number_of_features[0];j++)
				temp_data_matrix2[i][j]=h_data_matrix[i][j]+temp_data_matrix1[i][j];
		eigenvalue=(double *)malloc(sizeof(double)*number_of_features[0]);
		eigenvector=double_matrix_allocation(number_of_features[0],number_of_features[0]);
		eigenvalue_eigenvector(path,temp_data_matrix2,eigenvector,eigenvalue,number_of_features[0]);
		for(j=0;j<number_of_features[0];j++)
			for(k=0;k<number_of_new_features;k++)
			{
				if(eigenvector[j][k]>=0.000001)
					new_featureset.basis_vector_matrix[0].data_matrix[j][k]=eigenvector[j][k];
				else
					new_featureset.basis_vector_matrix[0].data_matrix[j][k]=0.000001;
			}
		for(j=0;j<number_of_new_features;j++)
			new_featureset.correlation[j]=sqrt(eigenvalue[j]);
		double_matrix_deallocation(eigenvector,number_of_features[0]);
		free(eigenvalue);
		for(i=1;i<number_of_modalities;i++)
			matrix_multiplication(multiplication_inverse_covariance_matrix2[regularization_pointer[t][i]][i-1].data_matrix,new_featureset.basis_vector_matrix[i-1].data_matrix,new_featureset.basis_vector_matrix[i].data_matrix,number_of_features[i],number_of_features[i-1],number_of_new_features);
		for(i=0;i<number_of_modalities;i++)
		{
			transpose_basis_vector_data_matrix=double_matrix_allocation(number_of_new_features,number_of_features[i]);
			matrix_transpose(new_featureset.basis_vector_matrix[i].data_matrix,transpose_basis_vector_data_matrix,number_of_new_features,number_of_features[i]);
			matrix_multiplication(transpose_basis_vector_data_matrix,zero_mean_matrix[i].data_matrix,canonical_variables_data_matrix,number_of_new_features,number_of_features[i],number_of_samples);
			for(j=0;j<number_of_samples;j++)
				for(k=0;k<number_of_new_features;k++)
					new_featureset.canonical_variables_matrix.data_matrix[j][k]+=canonical_variables_data_matrix[k][j];
			double_matrix_deallocation(transpose_basis_vector_data_matrix,number_of_new_features);
		}
		for(i=0;i<number_of_modalities;i++)
			lambda[t][i]=lambda_minimum+regularization_pointer[t][i]*delta;
		for(i=0;i<number_of_modalities;i++)
		{
			strcpy(filename,path);
			sprintf(temp,"temp_basis_vector%d_%d.txt",i+1,t+1);
			strcat(filename,temp);
			write_20_decimal_paces_file(filename,new_featureset.basis_vector_matrix[i].data_matrix,number_of_features[i],number_of_new_features);
		}
		strcpy(filename,path);
		sprintf(temp,"h_%d.txt",t+1);
		strcat(filename,temp);
		write_file(filename,h_data_matrix,number_of_features[0],number_of_features[0]);
		strcpy(filename,path);
		sprintf(temp,"temp_canonical_variables_%d.txt",t+1);
		strcat(filename,temp);
		write_output_file(filename,new_featureset.canonical_variables_matrix.data_matrix,class_labels,number_of_samples,number_of_new_features,number_of_class_labels);
		strcpy(filename,path);
		sprintf(temp,"temp_correlation_%d.txt",t+1);
		strcat(filename,temp);
		write_correlation_file(filename,new_featureset.correlation,number_of_new_features);
		strcpy(filename,path);
		sprintf(temp,"temp_lambda_%d.txt",t+1);
		strcat(filename,temp);
		write_lambda_file(filename,lambda,t+1,number_of_modalities);
		printf("\nNumber of Iteration = %d",t+1);
	}
	double_matrix_deallocation(temp_data_matrix1,number_of_features[0]);
	double_matrix_deallocation(temp_data_matrix2,number_of_features[0]);
	for(i=0;i<count;i++)
	{
		double_matrix_deallocation(inverse_covariance_matrix[i].data_matrix,number_of_features[0]);
		for(j=0;j<number_of_modalities-1;j++)
		{
			double_matrix_deallocation(multiplication_inverse_covariance_matrix1[i][j].data_matrix,number_of_features[0]);
			double_matrix_deallocation(multiplication_inverse_covariance_matrix2[i][j].data_matrix,number_of_features[j+1]);
		}
		free(multiplication_inverse_covariance_matrix1[i]);
		free(multiplication_inverse_covariance_matrix2[i]);
	}
	free(inverse_covariance_matrix);
	free(multiplication_inverse_covariance_matrix1);
	free(multiplication_inverse_covariance_matrix2);
	double_matrix_deallocation(h_data_matrix,number_of_features[0]);
	double_matrix_deallocation(canonical_variables_data_matrix,number_of_new_features);
	int_matrix_deallocation(regularization_pointer,combination);
	canonical_variables=(struct modality *)malloc(sizeof(struct modality)*number_of_new_features);
	for(i=0;i<number_of_new_features;i++)
		canonical_variables[i].data_matrix=double_matrix_allocation(number_of_samples,combination);
	for(i=0;i<combination;i++)
	{
		strcpy(filename,path);
		sprintf(temp,"temp_canonical_variables_%d.txt",i+1);
		strcat(filename,temp);
		read_input_canonical_file(filename,new_featureset.canonical_variables_matrix.data_matrix,class_labels);
		for(j=0;j<number_of_new_features;j++)
			for(k=0;k<number_of_samples;k++)
				canonical_variables[j].data_matrix[k][i]=new_featureset.canonical_variables_matrix.data_matrix[k][j];
	}
	number_of_merged_features=0;
	for(t=0;t<number_of_modalities;t++)
		number_of_merged_features+=number_of_features[t];
	new_data_matrix=double_matrix_allocation(number_of_merged_features,number_of_samples);
	k=0;
	for(t=0;t<number_of_modalities;t++)
	{
		for(i=0;i<number_of_features[t];i++)
		{
			for(j=0;j<number_of_samples;j++)
				new_data_matrix[k][j]=transpose_matrix[t].data_matrix[i][j];
			k++;
		}
	}
	if(k!=number_of_merged_features)
	{
		printf("\nError: Error in Computation.\n");
		exit(0);
	}
	zero_mean_data_matrix=double_matrix_allocation(number_of_merged_features,number_of_samples);
	zero_mean(new_data_matrix,zero_mean_data_matrix,number_of_merged_features,number_of_samples);
	covariance_data_matrix=double_matrix_allocation(number_of_merged_features,number_of_merged_features);
	within_set_covariance(zero_mean_data_matrix,covariance_data_matrix,number_of_merged_features,number_of_samples);
	eigenvalue=(double *)malloc(sizeof(double)*number_of_merged_features);
	eigenvector_data_matrix=double_matrix_allocation(number_of_merged_features,number_of_merged_features);
	eigenvalue_eigenvector(path,covariance_data_matrix,eigenvector_data_matrix,eigenvalue,number_of_merged_features);
	basis_vector_data_matrix=double_matrix_allocation(number_of_merged_features,number_of_new_features);
	copy_eigenvector(eigenvector_data_matrix,basis_vector_data_matrix,number_of_merged_features,number_of_new_features);
	k=0;
	for(t=0;t<number_of_modalities;t++)
	{
		for(i=0;i<number_of_features[t];i++)
		{
			for(j=0;j<number_of_new_features;j++)
				optimal_featureset.basis_vector_matrix[t].data_matrix[i][j]=basis_vector_data_matrix[k][j];
			k++;
		}
	}
	if(k!=number_of_merged_features)
	{
		printf("\nError: Error in Computation.\n");
		exit(0);
	}
	canonical_variables_data_matrix=double_matrix_allocation(number_of_new_features,number_of_samples);
	for(i=0;i<number_of_samples;i++)
		for(j=0;j<number_of_new_features;j++)
			optimal_featureset.canonical_variables_matrix.data_matrix[i][j]=0;
	for(t=0;t<number_of_modalities;t++)
	{
		transpose_basis_vector_data_matrix=double_matrix_allocation(number_of_new_features,number_of_features[t]);
		matrix_transpose(optimal_featureset.basis_vector_matrix[t].data_matrix,transpose_basis_vector_data_matrix,number_of_new_features,number_of_features[t]);
		matrix_multiplication(transpose_basis_vector_data_matrix,zero_mean_matrix[t].data_matrix,canonical_variables_data_matrix,number_of_new_features,number_of_features[t],number_of_samples);
		for(i=0;i<number_of_samples;i++)
			for(j=0;j<number_of_new_features;j++)
				optimal_featureset.canonical_variables_matrix.data_matrix[i][j]+=canonical_variables_data_matrix[j][i];
		double_matrix_deallocation(transpose_basis_vector_data_matrix,number_of_new_features);
	}
	double_matrix_deallocation(new_data_matrix,number_of_merged_features);
	double_matrix_deallocation(zero_mean_data_matrix,number_of_merged_features);
	double_matrix_deallocation(covariance_data_matrix,number_of_merged_features);
	double_matrix_deallocation(eigenvector_data_matrix,number_of_merged_features);
	double_matrix_deallocation(basis_vector_data_matrix,number_of_merged_features);
	double_matrix_deallocation(canonical_variables_data_matrix,number_of_new_features);
	free(eigenvalue);
	for(i=0;i<number_of_modalities;i++)
	{
		strcpy(filename,path);
		sprintf(temp,"basis_vector%d.txt",i+1);
		strcat(filename,temp);
		write_20_decimal_paces_file(filename,optimal_featureset.basis_vector_matrix[i].data_matrix,number_of_features[i],number_of_new_features);
	}
	strcpy(filename,path);
	strcat(filename,"canonical_variables.txt");
	write_output_file(filename,optimal_featureset.canonical_variables_matrix.data_matrix,class_labels,number_of_samples,number_of_new_features,number_of_class_labels);
	for(t=0;t<count;t++)
		for(i=0;i<number_of_modalities;i++)
		{
			if(i)
			{
				strcpy(filename,path);
				sprintf(temp,"multiplication_inverse_covariance_matrix1_%d_%d.txt",t,i-1);
				strcat(filename,temp);
				strcpy(temp,"rm -f ");
				strcat(temp,filename);
				system(temp);
				strcpy(filename,path);
				sprintf(temp,"multiplication_inverse_covariance_matrix2_%d_%d.txt",t,i-1);
				strcat(filename,temp);
				strcpy(temp,"rm -f ");
				strcat(temp,filename);
				system(temp);
			}
			else
			{
				strcpy(filename,path);
				sprintf(temp,"inverse_covariance_matrix_%d.txt",t);
				strcat(filename,temp);
				strcpy(temp,"rm -f ");
				strcat(temp,filename);
				system(temp);
			}
		}
	for(t=0;t<combination;t++)
	{
		for(i=0;i<number_of_modalities;i++)
		{
			strcpy(filename,path);
			sprintf(temp,"temp_basis_vector%d_%d.txt",i+1,t+1);
			strcat(filename,temp);
			strcpy(temp,"rm -f ");
			strcat(temp,filename);
			system(temp);
		}
		strcpy(filename,path);
		sprintf(temp,"h_%d.txt",t+1);
		strcat(filename,temp);
		strcpy(temp,"rm -f ");
		strcat(temp,filename);
		system(temp);
		strcpy(filename,path);
		sprintf(temp,"temp_canonical_variables_%d.txt",t+1);
		strcat(filename,temp);
		strcpy(temp,"rm -f ");
		strcat(temp,filename);
		system(temp);
		strcpy(filename,path);
		sprintf(temp,"temp_correlation_%d.txt",t+1);
		strcat(filename,temp);
		strcpy(temp,"rm -f ");
		strcat(temp,filename);
		system(temp);
		strcpy(filename,path);
		sprintf(temp,"temp_lambda_%d.txt",t+1);
		strcat(filename,temp);
		strcpy(temp,"rm -f ");
		strcat(temp,filename);
		system(temp);
	}
	for(i=0;i<number_of_new_features;i++)
	{
		for(j=0;j<number_of_modalities;j++)
		{
			strcpy(filename,path);
			sprintf(temp,"basis_vector%d_%d.txt",j+1,i+1);
			strcat(filename,temp);
			strcpy(temp,"rm -f ");
			strcat(temp,filename);
			system(temp);
		}
		strcpy(filename,path);
		sprintf(temp,"canonical_variables_%d.txt",i+1);
		strcat(filename,temp);
		strcpy(temp,"rm -f ");
		strcat(temp,filename);
		system(temp);
		strcpy(filename,path);
		sprintf(temp,"rhepm_%d.txt",i+1);
		strcat(filename,temp);
		strcpy(temp,"rm -f ");
		strcat(temp,filename);
		system(temp);
		strcpy(filename,path);
		sprintf(temp,"correlation_%d.txt",i+1);
		strcat(filename,temp);
		strcpy(temp,"rm -f ");
		strcat(temp,filename);
		system(temp);
		strcpy(filename,path);
		sprintf(temp,"relevance_%d.txt",i+1);
		strcat(filename,temp);
		strcpy(temp,"rm -f ");
		strcat(temp,filename);
		system(temp);
		strcpy(filename,path);
		sprintf(temp,"lambda_%d.txt",i+1);
		strcat(filename,temp);
		strcpy(temp,"rm -f ");
		strcat(temp,filename);
		system(temp);
	}
	for(i=0;i<number_of_modalities;i++)
	{
		double_matrix_deallocation(new_featureset.basis_vector_matrix[i].data_matrix,number_of_features[i]);
		double_matrix_deallocation(optimal_featureset.basis_vector_matrix[i].data_matrix,number_of_features[i]);
	}
	free(new_featureset.basis_vector_matrix);
	free(optimal_featureset.basis_vector_matrix);
	for(i=0;i<number_of_new_features;i++)
		int_matrix_deallocation(optimal_featureset.rhepm_data_matrix[i].data_matrix,number_of_class_labels);
	free(optimal_featureset.rhepm_data_matrix);
	free(optimal_featureset.relevance);
	double_matrix_deallocation(new_featureset.canonical_variables_matrix.data_matrix,number_of_samples);
	double_matrix_deallocation(optimal_featureset.canonical_variables_matrix.data_matrix,number_of_samples);
	free(new_featureset.correlation);
	free(optimal_featureset.correlation);
	for(i=0;i<number_of_new_features;i++)
		double_matrix_deallocation(canonical_variables[i].data_matrix,number_of_samples);
	free(canonical_variables);
	double_matrix_deallocation(lambda,combination);
	double_matrix_deallocation(optimal_lambda_objective_function_value,number_of_new_features);
	free(filename);
	free(temp);
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

void copy_eigenvector(double **eigenvector_data_matrix,double **basis_vector_data_matrix,int number_of_features,int number_of_new_features)
{
	int i,j;

	for(i=0;i<number_of_features;i++)
		for(j=0;j<number_of_new_features;j++)
			basis_vector_data_matrix[i][j]=eigenvector_data_matrix[i][j];
}

int rhepm(struct featureset optimal_featureset,double **canonical_variables_data_matrix,double **optimal_lambda_objective_function_value,int *class_labels,int number_of_modalities,int number_of_samples,int number_of_new_features,int combination,int number_of_class_labels,double epsilon,double omega)
{
	int i,j;
	int combination_number;
	double joint_dependency;
	double maximum_objective_function_value;
	double *each_feature_data_matrix;
	int **resultant_equivalence_partition;
	struct feature *new_feature;

	new_feature=(struct feature *)malloc(sizeof(struct feature)*combination);

	for(i=0;i<combination;i++)
	{
		new_feature[i].rhepm_data_matrix=int_matrix_allocation(number_of_class_labels,number_of_samples);
		each_feature_data_matrix=(double *)malloc(sizeof(double)*number_of_samples);
		for(j=0;j<number_of_samples;j++)
			each_feature_data_matrix[j]=canonical_variables_data_matrix[j][i];
		generate_equivalence_partition(new_feature[i].rhepm_data_matrix,each_feature_data_matrix,class_labels,number_of_samples,number_of_class_labels,epsilon);
		new_feature[i].relevance=dependency_degree(new_feature[i].rhepm_data_matrix,number_of_samples,number_of_class_labels);
		free(each_feature_data_matrix);
	}
	for(i=0;i<combination;i++)
	{
		new_feature[i].significance=0;
		for(j=0;j<number_of_new_features;j++)
		{
			resultant_equivalence_partition=int_matrix_allocation(number_of_class_labels,number_of_samples);
			form_resultant_equivalence_partition_matrix(new_feature[i].rhepm_data_matrix,optimal_featureset.rhepm_data_matrix[j].data_matrix,resultant_equivalence_partition,number_of_class_labels,number_of_samples);
			joint_dependency=dependency_degree(resultant_equivalence_partition,number_of_samples,number_of_class_labels);
			new_feature[i].significance+=joint_dependency-optimal_featureset.relevance[j];
			int_matrix_deallocation(resultant_equivalence_partition,number_of_class_labels);
		}
		if(number_of_new_features)
			new_feature[i].significance/=number_of_new_features;
		new_feature[i].objective_function_value=omega*new_feature[i].relevance+(1-omega)*new_feature[i].significance;
	}
	combination_number=0;
	maximum_objective_function_value=new_feature[0].objective_function_value;
	for(i=1;i<combination;i++)
		if(maximum_objective_function_value<new_feature[i].objective_function_value)
		{
			combination_number=i;
			maximum_objective_function_value=new_feature[i].objective_function_value;
		}
	optimal_featureset.relevance[number_of_new_features]=new_feature[combination_number].relevance;
	for(i=0;i<number_of_class_labels;i++)
		for(j=0;j<number_of_samples;j++)
			optimal_featureset.rhepm_data_matrix[number_of_new_features].data_matrix[i][j]=new_feature[combination_number].rhepm_data_matrix[i][j];
	optimal_lambda_objective_function_value[number_of_new_features][number_of_modalities]=maximum_objective_function_value;

	for(i=0;i<combination;i++)
		int_matrix_deallocation(new_feature[i].rhepm_data_matrix,number_of_class_labels);
	free(new_feature);

	return combination_number;
}

void generate_equivalence_partition(int **rhepm_data_matrix,double *data_matrix,int *class_labels,int number_of_samples,int number_of_class_labels,double epsilon)
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

void write_rhepm_file(char *filename,int **new_data_matrix,int row,int column)
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
			fprintf(fp_write,"%d\t",new_data_matrix[i][j]);
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

void write_relevance_file(char *filename,double *new_data_matrix,int size_of_matrix)
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
