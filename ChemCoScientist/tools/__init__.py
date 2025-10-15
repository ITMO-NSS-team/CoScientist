from langchain.tools.render import render_text_description
from protollm.tools.web_tools import web_tools

from ChemCoScientist.dataset_handler.chembl.chembl_utils import get_filtered_data
from ChemCoScientist.tools.chemist_tools import chem_tools, chem_tools_rendered, data_tools, data_tools_rendered
from ChemCoScientist.tools.nano_tools import nano_tools_rendered, nanoparticle_tools
from ChemCoScientist.tools.paper_analysis_tools import paper_analysis_tools, paper_analysis_tools_rendered

if web_tools:
    tools_rendered = render_text_description(
        web_tools + chem_tools + nanoparticle_tools + paper_analysis_tools
    ).replace("duckduckgo_results_json", "duckduckgo")
else:
    tools_rendered = render_text_description(chem_tools + nanoparticle_tools + paper_analysis_tools + data_tools)
