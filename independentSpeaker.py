from lxml import etree
import os

def create_participant_to_meeting_mapping(path):
    participant_to_meeting = {}
    
    dirname = os.path.dirname(__file__)
    xml_file_path = os.path.join(dirname, path)
    
    tree = etree.parse(xml_file_path)
    meetings = tree.xpath("//Meeting")
    
    for meeting in meetings:
        meeting_id = meeting.get("Session")
        
        participants = meeting.xpath("./Preamble/Participants/Participant[@Name and @Channel]")
        
        for participant in participants:
            participant_id = participant.get("Name")
            
            if participant_id not in participant_to_meeting:
                participant_to_meeting[participant_id] = []
                
            participant_to_meeting[participant_id].append(meeting_id)
    
    return participant_to_meeting

# Example usage
participant_to_meeting_mapping = create_participant_to_meeting_mapping("D:/linda/study/MINF2/data/ICSI_original_transcripts/transcripts/Bdb001.mrt")

# Print the resulting mapping
for participant_id, meeting_ids in participant_to_meeting_mapping.items():
    print(f" {participant_id} : {', '.join(meeting_ids)}")