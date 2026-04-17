class ProjectTool:

    def __init__(self):

        self.projects = {
            "mumbai bridge": {
                "contractor": "ABC Infra",
                "budget": "₹400 Cr",
                "status": "under construction"
            },
            "delhi metro": {
                "contractor": "DMRC",
                "budget": "₹1200 Cr",
                "status": "operational"
            },
            "pune highway": {
                "contractor": "XYZ Infra",
                "budget": "₹800 Cr",
                "status": "planning"
            }
        }

    def get_project_info(self, name):

        return self.projects.get(name.lower(), "project not found")
