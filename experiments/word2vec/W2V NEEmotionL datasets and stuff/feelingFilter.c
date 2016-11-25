#include<stdio.h>

#include<string.h>



void main(int argc, char ** argv)

{





FILE * f1;



f1= fopen("feelings-wordlist.txt","r");





FILE *f;



f=fopen("Filtered-Feelings.txt","w");





char tp[50];



while(!feof(f1))

{

fscanf(f1,"%s",tp);

fprintf(f,"%s\n",tp);

fscanf(f1,"%s",tp);

fscanf(f1,"%s",tp);

}





}