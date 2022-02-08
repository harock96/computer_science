package com.example.demo.controller;

public class Aveage {
	 
	   public Aveage() {
	       
	   }
	   public Aveage(Integer rating, float aveage) {
	      super();
	      this.rating = rating;
	      this.aveage = aveage;
	   }
	    
	   public Integer rating;
	   public float aveage;
	  
	   //getters and setters
	 
	   @Override
	   public String toString() {
	      return "Aveage [rating=" + rating + ", aveage=" + aveage + "]";
	   }
	}
