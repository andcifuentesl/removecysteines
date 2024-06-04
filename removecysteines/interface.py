import os
import ipywidgets as widgets
from IPython.display import display,clear_output

from . import removecysteines



def gui():
	out = widgets.Output()

	wl = widgets.Layout(width='80%',height='24pt')
	wl2 = widgets.Layout(width='80%',height='1in')
	wl3 = widgets.Layout(width='2in',height='0.25in')
	ws = {'description_width':'initial'}
	

	text_sequence = widgets.Textarea(placeholder='Enter amino acid sequence here....', description='WT Sequence:',layout=wl2, style=ws)
	text_name = widgets.Textarea(placeholder='Enter protein name here....', description='File Prefix:', layout=wl, style=ws)

	dropdown_model = widgets.Dropdown(options=removecysteines.ESM_models, description='ESM2 model:', value='esm2_t33_650M_UR50D', style=ws)
	dropdown_device = widgets.Dropdown(options=removecysteines.devices, description='Device:', style=ws)
	int_n_rounds = widgets.BoundedIntText(value=20,min=1,max=1000000, description='Maximum number of polishing steps:',style=ws)
	dropdown_noplots = widgets.Dropdown(value='False',options=['True','False'],ensure_option=True,description='Skip Plots?',style=ws)

	accordion_options = widgets.Accordion(children=[widgets.VBox([dropdown_model,dropdown_device,int_n_rounds,dropdown_noplots,]),],titles=('Options',),layout=widgets.Layout(width='80%'))
	button_run = widgets.Button(description="Run Optimization")

	vbox = widgets.VBox([text_sequence,text_name,accordion_options,button_run,])


	def click_run(b):
		### Parse input widget values
		wt_sequence = text_sequence.value
		ESM_model_name = dropdown_model.value
		device = dropdown_device.value
		n_rounds = int_n_rounds.value
		no_plots = dropdown_noplots.value == 'True'
		output_prefix = text_name.value

		# removecysteines.main(wt_sequence,ESM_model_name,device,n_rounds,no_plots,output_prefix,'./')
		removecysteines.main(wt_sequence,ESM_model_name,device,n_rounds,no_plots,output_prefix,None)

	def show_ui():
		with out:
			display(vbox)

	button_run.on_click(click_run)

	display(out)
	show_ui()



