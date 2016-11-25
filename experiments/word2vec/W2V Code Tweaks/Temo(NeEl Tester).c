//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include "csv2arr.h"
const long long max_size = 2000;         // max length of strings
const long long N = 40;                  // number of closest words that will be shown
const long long max_w = 50;              // max length of vocabulary entries

int main(int argc, char **argv) {
  FILE *f;
  char st1[max_size];
  char *bestw[N],*beste[N];
  char file_name[max_size], st[100][max_size];
  float dist, len, bestd[N], vec[max_size];
  long long words, size, a, b, c, d, cn, bi[100];
  char ch;
  float *M;
  char *vocab;
  if (argc < 2) {
    printf("Usage: ./distance <FILE>\nwhere FILE contains word projections in the BINARY FORMAT\n");
    return 0;
  }
  strcpy(file_name, argv[1]);
//bin file
  f = fopen(file_name, "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    return -1;
  }
  fscanf(f, "%lld", &words);  // no of words in bin file (vocabulary)
  fscanf(f, "%lld", &size);   //size of each word vector
  // has all words alone
  csv2arr("emotions.txt");
  vocab = (char *)malloc((long long)words * max_w * sizeof(char));
  for (a = 0; a < N; a++) bestw[a] = (char *)malloc(max_size * sizeof(char));
    for (a = 0; a < N; a++) beste[a] = (char *)malloc(max_size * sizeof(char));
  //M is the memory to store the vectors representing each word (float values list basically)
  M = (float *)malloc((long long)words * (long long)size * sizeof(float));
  if (M == NULL) {
    printf("Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)words * size * sizeof(float) / 1048576, words, size);
    return -1;
  }
  for (b = 0; b < words; b++) {
    a = 0;
    while (1) {
      // gets each word into vocab - b is the index of a word and a is to extract characters
      vocab[b * max_w + a] = fgetc(f);
      if (feof(f) || (vocab[b * max_w + a] == ' ')) break;
      if ((a < max_w) && (vocab[b * max_w + a] != '\n')) a++;
    }
    vocab[b * max_w + a] = 0;
    // the file has the weighted vector representation of each word following the word itself
    for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
    len = 0;
//calculating the magnitude of vector
    for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
    len = sqrt(len);
//calculating the part of cosine similarity for a word   
 for (a = 0; a < size; a++) M[a + b * size] /= len;
  }
  fclose(f);
  int p=0;
  while (1) {
    for (a = 0; a < N; a++) bestd[a] = 0;
    for (a = 0; a < N; a++) bestw[a][0] = 0;
          for (a = 0; a < N; a++) beste[a][0] = 0;
  
    a = 0;

    // repeat for each word in emotons.txts
    strcpy(st[0],csv2a.w[p]);
    if (csv2a.size==p) break;
    
    cn = 0;
    b = 0;
    c = 0;
    cn++;
    for (a = 0; a < cn; a++) {
      for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st[a])) break;
      if (b == words) b = -1;
      bi[a] = b;
      printf("\nWord: %s  Position in vocabulary: %lld\n", st[a], bi[a]);
      if (b == -1) {
        printf("Out of dictionary word!\n");
        csv2a.flag[p]=0;  
         break;
      }
      else
     csv2a.flag[p]=1;
    }
        printf("\n                                              Word       Cosine distance\n------------------------------------------------------------------------\n");
    for (a = 0; a < size; a++) vec[a] = 0;
    for (b = 0; b < cn; b++) {
      if (bi[b] == -1) continue;
      for (a = 0; a < size; a++) vec[a] += M[a + bi[b] * size];
    }
    len = 0;
    for (a = 0; a < size; a++) len += vec[a] * vec[a];
    len = sqrt(len);
    for (a = 0; a < size; a++) vec[a] /= len;
    
// own code

// vec of word or phrase stored  
    if(csv2a.flag[p]!=0)
    {
    for (a = 0; a < size; a++) csv2a.vec[p][a]=vec[a];
    for (a = 0; a < size; a++) printf("%f ",csv2a.vec[p][a]);
    }
  else
    for (a = 0; a < size; a++) csv2a.vec[p][a]=0.0;
    p++;

  }


int init = csv2a.size;
float learnR=0.50;

// training

FILE *training;
training = fopen("1.txt","r");
char t[100];
while(!feof(training))
{
    fscanf(training,"%s",t);
    int u=0;
    for (a = 0; a < N; a++) bestd[a] = 0;
    for (a = 0; a < N; a++) bestw[a][0] = 0;
          for (a = 0; a < N; a++) beste[a][0] = 0;
    a = 0;
    while (1) {
      st1[a] = t[u++];
      if ((st1[a] == '\n') || (a >= max_size - 1)) {
        st1[a] = 0;
        break;
      }
      a++;
    }
    if (!strcmp(st1, "EXIT")) break;
    cn = 0;
    b = 0;
    c = 0;
    while (1) {
      st[cn][b] = st1[c];
      b++;
      c++;
      st[cn][b] = 0;
      if (st1[c] == 0) break;
      if (st1[c] == ' ') {
        cn++;
        b = 0;
        c++;
      }
    }
    cn++;
    for (a = 0; a < cn; a++) {
      for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st[a])) break;
      if (b == words) b = -1;
      bi[a] = b;
      printf("\nWord: %s  Position in vocabulary: %lld\n", st[a], bi[a]);
      if (b == -1) {
        printf("Out of dictionary word!\n");  
        break;
      }
    }
    if (b == -1) continue;
    for (a = 0; a < size; a++) vec[a] = 0;
    for (b = 0; b < cn; b++) {
      if (bi[b] == -1) continue;
      for (a = 0; a < size; a++) vec[a] += M[a + bi[b] * size];
    }
    len = 0;
    for (a = 0; a < size; a++) len += vec[a] * vec[a];
    len = sqrt(len);
    for (a = 0; a < size; a++) vec[a] /= len;
 //   for (a = 0; a < size; a++) printf("\n%f",vec[a]);
    for (a = 0; a < N; a++) bestd[a] = -1;
    for (a = 0; a < N; a++) bestw[a][0] = 0;
          for (a = 0; a < N; a++) beste[a][0] = 0;
    for (c = 0; c < csv2a.size; c++) {
      
     if(csv2a.flag[c]!=0)
    {  a = 0;
      dist = 0;
      for (a = 0; a < size; a++) dist += vec[a] * csv2a.vec[c][a];
      for (a = 0; a < N; a++) {
        if (dist > bestd[a]) {
          for (d = N - 1; d > a; d--) {
            bestd[d] = bestd[d - 1];
            strcpy(bestw[d], bestw[d - 1]);
          }
          bestd[a] = dist;
          strcpy(bestw[a], csv2a.w[c]);
          strcpy(beste[a], csv2a.e[c]);
          break;
      }
      }
      }
    }
        for (a = 0; a < N; a++) printf("%30s\t%30s\t%f\n", bestw[a],beste[a], bestd[a]);
    if(bestd[0]>=learnR)
    {
      //expand emo vocab
      int fl=0;
for (c = 0; c < csv2a.size; c++) {
if(strcmp(csv2a.w[c],t)==0)
  {fl=1;break;}
}
      if(fl==0)
      {
      strcpy(csv2a.w[csv2a.size],t);
      strcpy(csv2a.e[csv2a.size],beste[0]);
      csv2a.flag[csv2a.size]=1;
      for (a = 0; a < size; a++) csv2a.vec[csv2a.size][a] = vec[a] ;
      csv2a.size++;
    }

if(csv2a.size-init >=500)
  {
    learnR+=0.05;
    init = csv2a.size;
    printf("\nHurrrayyyy!!!!\n");
    int dummy;
    scanf("%d",&dummy);
}
    }

}






















// mapping all emotions done

// Unknown words/phrases are to be mapped into emotions





while(1)
{
 




    for (a = 0; a < N; a++) bestd[a] = 0;
    for (a = 0; a < N; a++) bestw[a][0] = 0;
          for (a = 0; a < N; a++) beste[a][0] = 0;
    printf("Enter word or sentence (EXIT to break): ");
    a = 0;
    while (1) {
      st1[a] = fgetc(stdin);
      if ((st1[a] == '\n') || (a >= max_size - 1)) {
        st1[a] = 0;
        break;
      }
      a++;
    }
    if (!strcmp(st1, "EXIT")) break;
    cn = 0;
    b = 0;
    c = 0;
    while (1) {
      st[cn][b] = st1[c];
      b++;
      c++;
      st[cn][b] = 0;
      if (st1[c] == 0) break;
      if (st1[c] == ' ') {
        cn++;
        b = 0;
        c++;
      }
    }
    cn++;
    for (a = 0; a < cn; a++) {
      for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st[a])) break;
      if (b == words) b = -1;
      bi[a] = b;
      printf("\nWord: %s  Position in vocabulary: %lld\n", st[a], bi[a]);
      if (b == -1) {
        printf("Out of dictionary word!\n");
        break;
      }
    }
    if (b == -1) continue;
    printf("\n                                              Word       Cosine distance\n------------------------------------------------------------------------\n");
    for (a = 0; a < size; a++) vec[a] = 0;
    for (b = 0; b < cn; b++) {
      if (bi[b] == -1) continue;
      for (a = 0; a < size; a++) vec[a] += M[a + bi[b] * size];
    }
    len = 0;
    for (a = 0; a < size; a++) len += vec[a] * vec[a];
    len = sqrt(len);
    for (a = 0; a < size; a++) vec[a] /= len;
 //   for (a = 0; a < size; a++) printf("\n%f",vec[a]);
    for (a = 0; a < N; a++) bestd[a] = -1;
    for (a = 0; a < N; a++) bestw[a][0] = 0;
          for (a = 0; a < N; a++) beste[a][0] = 0;
    for (c = 0; c < csv2a.size; c++) {
      
     if(csv2a.flag[c]!=0)
    {  a = 0;
      dist = 0;
      for (a = 0; a < size; a++) dist += vec[a] * csv2a.vec[c][a];
      for (a = 0; a < N; a++) {
        if (dist > bestd[a]) {
          for (d = N - 1; d > a; d--) {
            bestd[d] = bestd[d - 1];
            strcpy(bestw[d], bestw[d - 1]);
          }
          bestd[a] = dist;
          strcpy(bestw[a], csv2a.w[c]);
          strcpy(beste[a], csv2a.e[c]);
          break;
      }
      }
      }
    }
    for (a = 0; a < N; a++) printf("%30s\t%30s\t%f\n", bestw[a],beste[a], bestd[a]);

}    

  return 0;
}

