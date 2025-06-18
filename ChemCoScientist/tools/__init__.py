from langchain.tools.render import render_text_description

from ChemCoScientist.tools.chemist_tools import chem_tools, chem_tools_rendered

from ChemCoScientist.tools.nano_tools import nano_tools_rendered, nanoparticle_tools
from protollm.tools.web_tools import web_tools
from ChemCoScientist.dataset_handler.chembl.chembl_utils import get_filtered_data

if web_tools:
    tools_rendered = render_text_description(
        web_tools + chem_tools + nanoparticle_tools
    )
else:
    tools_rendered = render_text_description(chem_tools + nanoparticle_tools)
