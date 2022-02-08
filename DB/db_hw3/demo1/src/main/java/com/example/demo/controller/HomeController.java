package com.example.demo.controller;
 
import java.util.ArrayList;
import java.util.List;
import java.sql.*;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
 
@RestController
public class HomeController {
    
	/*
    @RequestMapping(value="/")
    public String index() {
        
        return "index";
    }
    */
	@RequestMapping(value="/step1") //http://localhost:8000/employee
	 public List<Aveage> getAveage() throws SQLException 
	    {
	      List<Aveage> AveageList = new ArrayList<Aveage>();
	      
	      try (Connection conn = DriverManager.getConnection("jdbc:mariadb://localhost/hw2", "root","...")) // 비밀번호는변경.
	      {
	    	  try (Statement stmt = conn.createStatement())
	    	  {
	    		  String step1 = "select S.rating, avg(S.age) as avgage "
		        			+ "from Sailors S "
		        			+ "where S.age >= 18 "
		        			+ "group by S.rating "
		        			+ "having count(*)>1";
	    		  
	    		  
	    		  try (ResultSet rs = stmt.executeQuery(step1)) 
	    		  {
	    			  while (rs.next())
	    			  {
	    				  
	    				  AveageList.add(new Aveage(rs.getInt(1), rs.getFloat(2)));
	    				  
	    			  }
	    		  }
	    	  }
	      }
	      return AveageList;
	    }
}
	      
	    	
