//incomplete

#include<stdio.h>
#include<string.h>

void main()
{


//Stop Words

FILE * f2;

f2= fopen("Negative-StopWords.txt","r");

char negstop[16][20];
int i=0;

while(!feof(f2))
{
	fscanf(f2,"%s",negstop[i]);
printf("%s\n",negstop[i]);
        i++;
}


//Sentences Curator
FILE * f;
char file[1000000];
char temp[500];
char t;
i=0;
f= fopen("surprise.txt","r");

while(!feof(f))
{
t=fgetc(f);
file[i++]=t;
}


strcpy(temp,strtok(file,"\n"));
char *p;
while((p=strtok(NULL,"\n"))!=NULL)
{

        int k,m;
        char w[40];
        k=0; m=0;
        while(temp[m]!='\0')
{
if(temp[m]==' ')
{
if(chk(w))


k=0;
}
w[k++]=temp[m++];

}       
	//printf("%s",temp);
	strcpy(temp,p);
}

	
}
