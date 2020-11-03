import numpy as np
import pandas as pd
import pickle
import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import tkinter as tk
from tkinter.ttk import *

class gui_widget:
    def __init__(self):
        self.img_block_size=10 # Number of subplots per screen
        self.curr_block_size=self.img_block_size # Size of current block. Set in f_plot_imgs function
        self.iD=0
        #self.chart
#         self.value_lst=[tk.IntVar() for i in range(self.curr_block_size)]

    def f_plot_imgs(self,df,root):
        
        iD=self.iD
        if df.shape[0]<=(self.iD+self.img_block_size):
            self.curr_block_size=(df.shape[0]-iD)
        else:
            self.curr_block_size=self.img_block_size 
        
        self.value_lst=[tk.IntVar() for i in range(self.curr_block_size)]

        print("block_size",self.curr_block_size) 
        frame=Frame(root)
        frame.grid(row=0,column=1)
        
        fig,axarr=plt.subplots(self.curr_block_size,3,figsize=(3,self.curr_block_size), gridspec_kw = {'wspace':0, 'hspace':0},dpi=100)
        self.chart = FigureCanvasTkAgg(fig, root)
        keys=['temp','srch','diff']
        
        title="Plot indices: {0} to {1}.  ".format(iD,iD+self.curr_block_size-1)
        df_temp=df.iloc[range(iD,iD+self.curr_block_size-1)]
        
        if 1 not in df_temp.changed_label.values: # Only 0s
            header='Only Non-artifacts'
        elif 0 not in df_temp.changed_label.values: # Only 1s
            header='Only Artifacts'
        else : header= 'Both Artifacts and Non-artifacts'
        
        title+=header
        
        fig.suptitle(title)

        self.chart.get_tk_widget().grid(row=0,column=0)
        dict1={'1':'Artifact','0':'Non-artifact'}
        label=dict1[str(df.iloc[iD]['changed_label'])]

        for row in np.arange(self.curr_block_size):
            for col in range(3):
                if self.curr_block_size==1: axarr=np.reshape(axarr,(self.curr_block_size,3)) ## Exception handling for 1 image in block
                if row==0: axarr[row,col].set_title(keys[col])
                axarr[row,col].imshow(df.iloc[iD+row].imgs[col,:,:], origin='lower',  extent = [0, 51, 0, 51])
#            self.chart.draw()

            ## Create flip button
            self.flip_btn=tk.Checkbutton(frame,text='%s Labeled as: %s.\nChange?'%(str(row+1),label),variable=self.value_lst[row],onvalue =1,offvalue=0,width=30,height=4,pady=6,padx=1,bd=5,bg='grey',fg='Blue',justify='left')
            self.flip_btn.grid(row=row,column=3)

        temp=plt.setp([a.get_xticklabels() for a in axarr[:-1,:].flatten()], visible=False)
        temp=plt.setp([a.get_yticklabels() for a in axarr[:,1:].flatten()], visible=False)
        
        plt.close()


    def f_next(self,df,root):
        
#        vals=np.array([int(val.get()) for val in self.value_lst]) ## Store all button variables
#        id_list=np.where(vals==1)[0] ## Pick indices where the button was clicked
#        print("Modified IDs",id_list) 

        for count,idx in enumerate(self.value_lst):
            val=int(idx.get())
            ID=self.iD+count ## Pick the right indices for dataframe
            if val==1: ## If button was clicked, flip label
                df.iloc[ID]['final_label']=(df.iloc[ID]['changed_label']+1)%2
            elif val==0 :   ## Store old label
                df.iloc[ID]['final_label']=df.iloc[ID]['changed_label']

        ## Move to next batch
        self.iD+=self.curr_block_size ## Update iD for next batch
        print("new ID: ",self.iD)
        
        if self.iD>=df.shape[0]: 
            print("All images checked.\nDone!")
            root.destroy()
            return
       
        ### Clear plots and buttons 
        self.flip_btn.grid_forget()
        self.chart.get_tk_widget().grid_forget()
        
        ### Run next batch of images
        self.f_plot_imgs(df,root)

if __name__=="__main__":
    
    ##################################
    ## Read data
    df_original=pd.read_pickle('df_images_labels.pkle')
    df=df_original.copy()
#    df=df.head(30)
#    df=df.sample(30)
     
    print("Total size",df.shape)
    save_cols=['ID','old_label','changed_label','final_label']

    ## Read results and pick IDs that remain. Prepare results dataframe
    fname='results.csv'
    if os.path.exists(fname):
        df_results_old=pd.read_csv(fname) 
        checked_IDs=df_results_old.ID.values
        unchecked_IDs=[i for i in df.ID.values if i not in checked_IDs]  
        if len(unchecked_IDs)<1: 
            print('Done. All images relabelled. To redo, delete the file `results.csv`')
            raise SystemExit

        ## Only get IDs that haven't been checked
        df=df[df.ID.isin(unchecked_IDs)] 
        print('Size of remaining data',df.shape)
    else:
        df_results_old=pd.DataFrame(columns=save_cols)
        print("Starting Fresh")
    
    df=df.sort_values(by='changed_label')
    
    ## 
    root=tk.Tk()
    root.title("Label Images") 
    root.geometry("1200x1000")

    ### Create object
    obj=gui_widget()

    ## Plot figures
    obj.f_plot_imgs(df,root)

    ## Proceed to next iteration
    obj.next_btn=Button(root,text='Save.Proceed to Next batch',command=lambda: obj.f_next(df,root))
    obj.next_btn.grid(row=0,column=4)

    ## Quit loop
    Button(root,text='Quit without saving.',command=root.destroy).grid(row=0,column=6)

    root.mainloop() 

    ### Save file 
    print("Saving results to file")
    df_results_new=df[df.final_label.isin([0,1])][save_cols]
#    df_results_new.to_pickle('df_results_new.pkle')
#    df_results_old.to_pickle('df_results_old.pkle')
    df_result=df_results_old.append(df_results_new)
    print("Previously completed relabeling size",df_results_old.shape)
    print("Current relabelling size",df_results_new.shape)
    print("Total results size",df_result.shape)
#    df_result.to_pickle('df_results.pkle')
    df_result.to_csv('results.csv',index=False,header=True)


