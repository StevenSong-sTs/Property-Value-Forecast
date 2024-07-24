"""
Purpose of this class is to fetch New Housing Unit (NHU) data from the Census
Data Base at the metropolitan area level. The current API does not support this
level of granualarity so this class softscrapes the available files.
"""

import os
from os.path import join
import sys
import requests  # type: ignore
import pandas as pd # type: ignore
import re
import  numpy as np # type: ignore
import argparse
from log_stdout import StreamToLogger

current_directory = os.getcwd()

stream_logger  = StreamToLogger()
    #log_file= os.path.join(current_directory,'assets/Logs/nhu_data_pull.log')
#)

logger = stream_logger.logger


defaults = {
    "start_year" : 2000,
    'end_year' : 2024,
    'base_url' : "https://www.census.gov/construction/bps",
    'txt_url_suffix' : "/txt/tb3u",
    'xls_url_suffix_former' : "/xls/msamonthly_",
    'xls_url_suffix_new' : "/xls/cbsamonthly_",    
    'save_path' : os.path.join(current_directory,
                                'assets/new_housing_units_metro'),
    'data_path' : os.path.join(current_directory,
                                'assets/Data')
    }


class create_nhu_data():
    def __init__(self, **kwargs):
        """
        sets up relevant pathing variables for fetching Census data  
        """
        self.start_year = kwargs.get("start_year", defaults["start_year"])
        self.end_year = kwargs.get("end_year", defaults["end_year"])
        self.base_url = kwargs.get("base_url", defaults["base_url"])
        self.txt_url_suffix = kwargs.get("txt_url_suffix", defaults["txt_url_suffix"])
        self.xls_url_suffix_former = kwargs.get("xls_url_suffix_former", defaults["xls_url_suffix_former"])
        self.xls_url_suffix_new = kwargs.get("xls_url_suffix_new", defaults["xls_url_suffix_new"])
        self.save_path = kwargs.get("save_path", defaults["save_path"])
        self.data_path = kwargs.get("data_path", defaults["data_path"])
        self.txt_ext=".txt"
        self.xls_ext=".xls"

        self.months = [f"{i:02d}" for i in range(1, 13)]
        self.month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        os.makedirs(self.save_path, exist_ok=True)
        self.__main()

    def __construct_links(self):
        """
        formats for the files changed year over year and as such so did the 
        download links. As we iterate over the years and months we account for
        these formats and construct the appropriate links
        """
        logger.info('Contructing Links')

        file_urls=[]
        file_date_info=[]    
        for year in range(self.start_year, self.end_year + 1):                  
            for month in self.months:
                if year < 2004:                    
                    file_url = f"{self.base_url}{self.txt_url_suffix}{str(year)[-2:]}{month}{self.txt_ext}"
                elif year <2019 or (year == 2019 and int(month)<=10):
                    #November 2019 the transitioned to xls                   
                    file_url = f"{self.base_url}{self.txt_url_suffix}{year}{month}{self.txt_ext}"
                elif year < 2024:
                    file_url = f"{self.base_url}{self.xls_url_suffix_former}{year}{month}{self.xls_ext}"
                else: 
                    file_url = f"{self. base_url}{self.xls_url_suffix_new}{year}{month}{self.xls_ext}"
                file_urls.append(file_url)
                file_date_info.append((year,month))
        
        logger.info('Links Constructed')
        
        return file_urls, file_date_info
    
    def __download_files(self):
        """
        fetches the txt and xls metro files via url link constructs, and
        downloads them to the specified save path in __init__
        """ 
        logger.info("Initializing download of metro data for new housing units")  
        file_urls, file_date_info = self.__construct_links()        

        for i, file in enumerate(file_urls):
            year=file_date_info[i][0]
            month=file_date_info[i][1]
            if file[-4:] == ".txt":
                file_name = f"nhu_metro_{year}_{month}{self.txt_ext}"
            else:
                file_name = f"nhu_metro_{year}_{month}{self.xls_ext}"

            file_path = os.path.join(self.save_path, file_name)
            if os.path.exists(file_path):
                logger.info(f'Already downloaded {file_name}')
                continue
                    
            response = requests.get(file)
            
            if response.status_code == 200:               
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Downloaded: {file_path}")  
            else:
                logger.error(f"Failed to download: {file_name}")
        return logger.info("new housing units data for metros downloaded")
    
    def __parse_txt(self, file_path):
        """
        takes one of the downloaded txt files and parses the data into a
        Pandas dataframe.  involves some cleaning and naming conventions
        """
       
        logger.info(f"Parsing {os.path.basename(file_path)}")
        with open(file_path, 'r') as file:            
            lines = file.readlines()           
            year = file_path[-11:-7]
            month = file_path[-6:-4]
            
            #filter out non data lines that contain whitespace, hyphens and 
            #asterisks. stips trailing and leading whitespace from lines
            lines=[line.strip() for line in lines if not re.match(
                r'\s*[-*\s]*$', line)
                ]
            
            # Find the start index
            for i, line in enumerate(lines):
                if 'Abilene' in line:
                    start_index = i
                    break

            # Find the end index
            for i, line in enumerate(lines):
                if 'Yuma' in line:
                    end_index = i

            lines=lines[start_index:end_index+1] 

            #Some names have return charaters within them creating excess rows
            #This loop catches these cases and merges the lines back together
            merged_lines = []
            last_line = ""             
            for line in lines:
                line = line.strip().replace('*', '')    
                if line == last_line:
                    #remove duplicates
                    continue      
                            
            

                if re.search(r'[A-Za-z,\-]$', last_line) != None:                    
                    if line[0].isnumeric():
                        line = last_line + "    " + line
                    else:
                        line = last_line + " " + line                    
                    last_line=line
                    del merged_lines[-1]
                    merged_lines.append(line)
                    continue
                    
                last_line=line
                merged_lines.append(line)        

                

            data=[]
            for line in merged_lines:                
                if line.strip() == '':
                    #double check for blank lines
                    continue 

                
                if int(year) > 2009 or (int(year)==2009 and int(month)>=10) or (int(year)==2009 and int(month)==8):
                    #new preceding columns added aug 2009 need to account for them
                    i = 2
                else:
                    i = 0              

                #expected that columns are seperated by at least 4 spaces
                #where this is not the case pad the numbers
                line = re.sub(r'(\d+)', r'    \1    ', line)
                columns = re.split(r'\s{4,}', line.strip())               

                #parse data into corresponding column and append data
                row = {
                    'Metropolitan Area': columns[i],
                    'Total': columns[i+1],
                    '1 Unit': columns[i+2],
                    '2 Units': columns[i+3],
                    '3 and 4 Units': columns[i+4],
                    '5 Units or More': columns[i+5],
                    'Structures with 5 Units or More': columns[i+6],            
                    'Year': year,
                    'Month': month
                }
                data.append(row)
            file.close()

        #return the data as a pandas dataframe        
        df=pd.DataFrame(data)     
        logger.info(f"Parsed {os.path.basename(file_path)}")        
        return df
    
    def __parse_xls(self, file_path):
        """
        takes one of the downloaded xls files and parses the data into a
        Pandas dataframe.  involves some cleaning and naming conventions
        """
        logger.info(f"Parsing {os.path.basename(file_path)}")

        #sheet 0 has the unit data which we want sheet 1 has the valuation
        df=pd.read_excel(file_path, sheet_name=0)

        #data has Year to Date columns which we don't need so drop

        
        num_columns=len(df.columns)
        start_col_index = 10
        ytd_columns=[int(num) for num in np.linspace(start_col_index ,num_columns-1,num_columns-start_col_index)]
        df.drop(df.columns[ytd_columns],axis=1,inplace=True)

        year = file_path[-11:-7]
        month = file_path[-6:-4]

        #actual column names are contained in row 6. extract and map rename
        #columns. update columns for consitency with __parse_txt remove header
        #rows
        column_mapping = dict(
            zip(df.columns.to_list(),df.iloc(0)[6].to_list())
            )
        df.rename(columns= column_mapping, inplace=True)
        df.rename(columns={
            'Name':'Metropolitan Area',
            'Num of Structures With 5 Units or More':
            'Structures with 5 Units or More'},
             inplace=True)
        df = df[['Metropolitan Area', 'Total', '1 Unit', '2 Units', '3 and 4 Units', '5 Units or More', 'Structures with 5 Units or More']]
        first_row =df[df['Metropolitan Area'].str.contains("Abilene", na=False)].index[0]
        df= df.iloc[first_row:].reset_index(drop=True)
        df['Metropolitan Area'] = df['Metropolitan Area'].str.strip()

        #Add date fields       
        df['Year']=year
        df['Month'] = month

        logger.info(f"Parsed {os.path.basename(file_path)}")
        return df
    
    def __process_files(self, files):
        """
        Given a list of files from a directory iterate over files and pass to
        apropriate parser based on extentsion. Returns the concatenation of all
        parsed dataframes 
        """
        logger.info("Begin file concatenation")
        df_list =[]        
        for file in files:
            file_path = os.path.join(self.save_path, file)
            file_ext = file[-4:]
            if file_ext == ".txt":
                df_list.append(self.__parse_txt(file_path))
            else:
                df_list.append(self.__parse_xls(file_path))

        df=pd.concat(df_list, ignore_index=True)
        logger.info("Data concatenated")
        return df
    
    def __main(self):
        logger.info("Initializing creation of NHU data set")

        #check if nhu_data.json exists and end program if true
        data_file_name = 'nhu_data.json'
        data_path = os.path.join(self.data_path,data_file_name)
        if os.path.exists(data_path):
            logger.info(f'{data_file_name} already exists')
            return
        
        self.__download_files()
        files=os.listdir(
            self.save_path
            )       
        df = self.__process_files(files)
        df.to_json(data_path, orient='records', lines=True)
        logger.info(f"{data_file_name} Created")
        return
       
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Set urls and save path")
    parser.add_argument('--base_url', type=str,
                        help='The base url for bps data')
    parser.add_argument('--txt_url_suffix', type=str,
                        help='url suffix to nav to pre 2020 txt files')
    parser.add_argument('--xls_url_suffix_former', type=str,
                        help='url suffix to nav to pre 2024 xls files')
    parser.add_argument('--xls_url_suffix_new', type=str,
                        help='url suffix to nav to post 2024 xls files')
    parser.add_argument('--start_year', type=str,
                        help='first year to begin pulling data from')
    parser.add_argument('--end_year', type=str,
                        help='last year to pull data from')
    parser.add_argument('--save_path', type=str,
                        help='destination folder path for downloads')
    parser.add_argument('--data_path', type=str,
                        help='destination folder path for final table')
    args = parser.parse_args()

    

    config={}
    for default in list(defaults.keys()):
        if getattr(args, default):
            config[default] = getattr(args, default)
        else:
            config[default] = defaults[default]

    create_nhu_data(**config) 
    


