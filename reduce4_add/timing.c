/*
 *  timing.c
 * 
 *
 *
 */


#include "timing.h"
#include <sys/time.h>
#include <stdlib.h>
#include <time.h>

float timeCost(clock_t start, clock_t end)
{
    return (float)(end - start)/CLOCKS_PER_SEC;
}

/* Subtract the `struct timeval' value 'then' from 'now',
   returning the difference as a float representing seconds
   elapsed.
*/
float elapsedTime(struct timeval now, struct timeval then){

   // Based on code from the gnu documentation.

   /* Perform the carry for the later subtraction by updating then. */
   if (now.tv_usec < then.tv_usec) {
   
      int nsec = (then.tv_usec - now.tv_usec) / 1000000 + 1;
      then.tv_usec -= 1000000 * nsec;
      then.tv_sec += nsec;
   }


   if (now.tv_usec - then.tv_usec > 1000000) {
   
      int nsec = (then.tv_usec - now.tv_usec) / 1000000;
      then.tv_usec += 1000000 * nsec;
      then.tv_sec -= nsec;
   }


   /* Compute the time remaining to wait.
     tv_usec is certainly positive. */
   int tv_sec = now.tv_sec - then.tv_sec;
   int tv_usec = now.tv_usec - then.tv_usec;
  
  
   return tv_sec + tv_usec / 1000000.0;
}


