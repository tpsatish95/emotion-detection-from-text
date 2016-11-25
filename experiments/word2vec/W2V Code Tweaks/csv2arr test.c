#include<stdio.h>
#include<string.h>
struct csv
{

char w[2000][100];
char e[2000][100];
}csv2a;

void main()

{
FILE * f;

f= fopen("emotions.txt","r");
int i=0;
while(!feof(f))
{
char temp[100];
	fscanf(f,"%s",temp);
strcpy(csv2a.w[i],strtok(temp,","));
strcpy(csv2a.e[i],strtok(NULL,","));        
printf("%s\t%s\n",csv2a.w[i],csv2a.e[i]);
    i++;
}


}
