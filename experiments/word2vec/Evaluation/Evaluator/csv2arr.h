//csv2arr header file



struct csv

{
char w[100000][100];
char e[100000][100];
float vec[100000][2000];
int size;
int flag[100000];

}csv2a;



void csv2arr(char file[])

{

FILE * f;



f= fopen(file,"r");

int i=0;

while(!feof(f))

{
char temp[100];

	fscanf(f,"%s",temp);
strcpy(csv2a.w[i],strtok(temp,","));
strcpy(csv2a.e[i],strtok(NULL,","));        


    i++;}


csv2a.size=i;


}
