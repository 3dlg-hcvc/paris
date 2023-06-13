import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
import json

def rewrite_json_from_urdf(model_id, download=False):
    if download:
        import sapien
        token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImpsYTg2MUBzZnUuY2EiLCJpcCI6IjE3Mi4yMC4wLjEiLCJwcml2aWxlZ2UiOjEsImZpbGVPbmx5Ijp0cnVlLCJpYXQiOjE2NjY0NjMyNjUsImV4cCI6MTY5Nzk5OTI2NX0.X_z53KjP3B7Pejd4hH2RTVflvw5uH2x88NXwv7HiHcs'
        urdf_file = sapien.asset.download_partnet_mobility(model_id, token, directory=os.path.join(ROOT_DIR, 'data/PartNet-Mobility'))
    else:
        urdf_file = os.path.join(ROOT_DIR, 'data/PartNet-Mobility', model_id, 'mobility.urdf')

    from lxml import etree as ET
    tree = ET.parse(urdf_file)
    root = tree.getroot()
    visuals_dict = {}
    for link in root.iter('link'):
        meshes = []
        for visuals in link.iter('visual'):
            meshes.append(visuals[1][0].attrib['filename'])
        visuals_dict.update({link.attrib['name']: meshes})
    
    # load .json file as a dict
    with open(os.path.join(ROOT_DIR, 'data/PartNet-Mobility', model_id, 'mobility_v2.json'), 'r') as f:
        meta = json.load(f)
        f.close()
    
    # find mesh files in urdf and add to meta
    for entry in meta:
        link_name = 'link_{}'.format(entry['id'])
        entry['visuals'] = visuals_dict[link_name]
    
    # write a self-used json file
    with open(os.path.join(ROOT_DIR, 'data/PartNet-Mobility', model_id, 'mobility_v2_self.json'), 'w') as json_out_file:
        json.dump(meta, json_out_file)
        json_out_file.close()
    

model_id = '9996'
rewrite_json_from_urdf(model_id, download=True)