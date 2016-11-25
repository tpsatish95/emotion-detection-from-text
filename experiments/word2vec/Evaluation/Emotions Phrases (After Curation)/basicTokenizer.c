//Incomplete

// See java Implentation

#include<stdio.h>
#include<string.h>

void main()
{


//Stop Words

FILE * f1;

f1= fopen("stop.txt","r");

char stop[174][20];
int i=0;

while(!feof(f1))
{
	fscanf(f1,"%s",stop[i]);
//        printf("%s\n",stop[i]);
        i++;
}



//Sentences Tokenize
FILE * f;

f= fopen("joy.txt","r");

while(!feof(f))
{
	char temp[100];
	fscanf(f,"%s",temp);
//	printf("%s\n",temp);


}
	
}
