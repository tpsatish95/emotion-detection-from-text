import java.io.*;
import java.util.*;
public class SentenceMain {

	public static void main(String args[])
	{
		List<String> a = null;
		int emoNUM=0;
		FileInputStream stop=null;
		FileInputStream emo[]= new FileInputStream[6];
		FileOutputStream emoOUT[] = new FileOutputStream[6];
		
		String punc[]={"1","2","3","4","5","6","7","8","9","0",",",":","(",")","[","]",".","-","!","?","\""};
		
		try {
			

		emo[0] = new FileInputStream("JOY.txt");
		 emo[1] = new FileInputStream("ANGER.txt");
		 emo[2] = new FileInputStream("LOVE.txt");
		 emo[3] = new FileInputStream("SADNESS.txt");
		 emo[4] = new FileInputStream("SURPRISE.txt");
		 emo[5] = new FileInputStream("FEAR.txt");
		 
		 emoOUT[0] = new FileOutputStream("joyO.txt");
		 emoOUT[1] = new FileOutputStream("angerO.txt");
		 emoOUT[2] = new FileOutputStream("loveO.txt");
		 emoOUT[3] = new FileOutputStream("sadnessO.txt");
		 emoOUT[4] = new FileOutputStream("surpriseO.txt");
		 emoOUT[5] = new FileOutputStream("fearO.txt");
		 
		 stop = new FileInputStream("stop.txt");
		 
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			System.out.println("File Not Found");
		}
		String emotions[] = {"joy","anger","love","sadness","surprise","fear"};
		
///		String negs[]={"cannot","do not","fail","little value","mistake","not","problem","unable to","unfortunately","don't","can't","i","feel"};
		
		String sp[]= new String[176];
		int o=0;
		Scanner sin = new Scanner(stop);
		while(sin.hasNextLine())
		{				
				sp[o++] = sin.nextLine();
				
		}
		System.out.println("Stop WOrds Over!!");
		
		while(emoNUM<=5)
		{
		
			Scanner in = new Scanner(emo[emoNUM]);
			
			String temp;
			
			while(in.hasNextLine())
			{
				
				try
				{
				temp= in.nextLine();
		
				//Twokenize
				try {
					 a= twokenize.retTOK(temp);
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				
				//remove stop words
				for (int i=0; i<a.size(); i++) {
	    		int f=0,f1=0,f2=0;
					String tt =	a.get(i);
	    		
	    		for(int k=0;k<sp.length;k++)
	    		{
	    			if(sp[k].trim().compareToIgnoreCase(tt.trim())==0)
	    			{
	    				f=1;
	    				break;
	    			}
	    		}
	    		
	    		if(f==1)
	    		{a.remove(i);	continue;}
	    		
	    		for(int y=0;y<punc.length;y++)
	    		{
	    			if(tt.contains(punc[y]))	
		    		{f1=1; break;}	
	    		}
	    		
	    		if(f1==1)
	    		{a.remove(i);	continue;}
	    		
	    			if(tt.trim().contains("feel"))	
		    		{f2=1;}	
	    		
	    		if(f2==1)
	    		{a.remove(i);	continue;}
	    		
	    		
	    	  //  if(emoNUM==4)
	    	    //	if(tt.toLowerCase().contains("surpris"))
	    	    	//{a.remove(i);continue;}
	    		
                // Feed in to word2Vec in C
				
				// TOo time consuming soooooo just store output in file
				
				}
				
				for(int i=0;i<a.size();i++)
				{
					emoOUT[emoNUM].write(a.get(i).getBytes());
					emoOUT[emoNUM].write(" ".getBytes());
				}
				
				emoOUT[emoNUM].write("\n".getBytes());				
				
				
				}//try
				catch(Exception e)
				{
					System.out.println("End of FILE");
				}
			}
			try {
				emoOUT[emoNUM].flush();
				emoOUT[emoNUM].close();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

			emoNUM++;
		}//while
	}
}
