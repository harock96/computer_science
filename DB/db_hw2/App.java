package hongik.ce.dbms;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.sql.*;

public class App {
	
	public static void main(String[] args) throws SQLException {

	//create connection for a server installed in localhost, with a user "root" with no password
	    try (Connection conn = DriverManager.getConnection("jdbc:mariadb://localhost/hw2", "root",'j02030339...')) { //비밀번호 부분은 변경
	        // create a Statement
	        try (Statement stmt = conn.createStatement()) {
	        	
	        	String step1 = "select S.rating, avg(S.age) as avgage "
	        			+ "from Sailors S "
	        			+ "where S.age >= 18 "
	        			+ "group by S.rating "
	        			+ "having count(*)>1";
	        	
	        	String step2 = "update sailors set age = 30 where sname = 'Andy'";
	        	
	        	HashMap<Integer,Float> map = new HashMap();
	        	HashMap<Integer,Float> map2 = new HashMap();
	  
	            //execute query
	            try (ResultSet rs = stmt.executeQuery(step1)) {
	            	
	            	System.out.println("rating\taveage");
	                
	                while(rs.next()) {
	     
		                System.out.print(rs.getInt(1)); 
		                System.out.println("\t" + rs.getFloat(2));
		                map.put(rs.getInt(1), rs.getFloat(2));
		                }  
	            }
	            
	            try (ResultSet rs2 = stmt.executeQuery(step2)){
	            }
	            
	            try (ResultSet rs3 = stmt.executeQuery(step1)) {
	            	
	            	
	                while(rs3.next()) {
	                	map2.put(rs3.getInt(1), rs3.getFloat(2));
	                	
	                	
	                }
	            }
		                    
	
	            
	            for (Integer key : map.keySet()) {
	           
	            	if ((double)map.get(key)!=(double)map2.get(key)) {
	            	
	            		System.out.println("Changed rating : " + key);
	            		
	            	}   	
	            }
	        }
	    }
	}
}

