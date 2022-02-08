#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "memory.h"

void print_name(struct record *);
void print_number(struct record *);
void print_data(char *, int);

// comparison function for records
int compare(char key[3], struct record *);

// data
struct record * data = NULL; // Initially NULL.

void init()
{
  init_pool();
}

void add(char *name, char *number)
{
  struct record *new;
  new = new_node();
  struct record *cur = data;
  struct record *pre = NULL;

  if (new == NULL)
  {
    printf("Can't add.  The address book is full!\n");
    return;
  }

  (new->name)[0] = name[0];
  (new->name)[1] = name[1];
  (new->name)[2] = name[2];
  (new->number)[0] = number[0];
  (new->number)[1] = number[1];
  (new->number)[2] = number[2];
  (new->number)[3] = number[3];
  
  if (data == NULL) 
  {
    new->next = NULL;
    data = new;
    return;
  }

  while (compare(name, cur) > 0)
  {
    pre = cur;
    cur = cur->next;
  }

  if (pre==NULL) {
    new->next = data;
    data = new;
    return;
  }
  pre->next = new;
  new->next = cur;
}


/* Just a wrapper of strncmp(), except for the case r is NULL.
Regard strncmp(a,b) as a-b, that is,
Negative value if key is less than r.
​0​ if key and r are equal.
Positive value if key is greater than r. */

int compare(char key[3], struct record *r)
{
  if (r==NULL)
    return -1;
  else {
    return strncmp(key, r->name, 3);
  }
}


void search(char name[3])  
{
  struct record *r=data;
  int result;
  while(r!=NULL && (result=compare(name,r))!=0)
    r=r->next;
  if(r==NULL)
    printf("Couldn't find the name.\n");
  else {
    print_name(r);
    printf(" : ");
    print_number(r);
    printf(" was found.\n");
  }
}


void delete(char name[3])
{
  struct record *cur=data;
  struct record *pre=NULL;

  if (cur == NULL)
  {
    printf("Couldn't find the name.\n");
    return;
  }

  while (compare(name, cur) != 0)
  {
    pre = cur;
    cur = cur->next;
  }

  if (pre == NULL)
  {
    data = data->next;
  }
  else
  {
    pre->next = cur->next;
  }
  free_node(cur);
  printf("The name was deleted.\n");
}


/* Just a wrapper of strncmp(), except for the case r is NULL. 
Regard strncmp(a,b) as a-b, that is,
Negative value if key is less than r.
​0​ if key and r are equal.
Positive value if key is greater than r. */

// Prints ith name.
void print_name(struct record *r)
{
  print_data(r->name, 3);
}

// Prints ith number.
void print_number(struct record *r)
{
  print_data(r->number, 4);
}

void print_data(char * s, int n)
{
  int i;
  for (i=0; i<n; i++)
    putchar(s[i]);
}

void print_list()
{
  struct record *cur=data;

  while (cur!=NULL) {

    print_name(cur);
    printf(" : ");
    print_number(cur);
    printf("\n");
    cur=cur->next;
  }
}

