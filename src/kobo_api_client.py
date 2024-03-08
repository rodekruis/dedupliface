import requests
import json

class KoboAPI:
    """
    Kobo API client.
    """
    
    def __init__(
            self,
            url: str,
            token: str,
            asset: str,
            submission: dict | None = None
    ):
        self.url = url
        self.token = token
        self.asset = asset
        self.data = None
        self.attachments = None
        if submission:
            self.process_submission(submission)
    
    def get_kobo_attachment(self, field):
        """Get attachment from kobo URL"""
        kobo_value_url = self.data[field].replace(" ", "_")
        file_url = self.attachments[kobo_value_url]['url']
        data_request = requests.get(file_url, headers={'Authorization': f'Token {self.token}'})
        data = data_request.content
        return data
    
    def process_submission(self, attachment_data):
        kobo_data_clean = {k.lower(): v for k, v in attachment_data.items()}
        # remove group names
        for key in list(kobo_data_clean.keys()):
            new_key = key.split('/')[-1]
            kobo_data_clean[new_key] = kobo_data_clean.pop(key)
        self.data = kobo_data_clean
        # create a dictionary that maps the attachment filenames to their URL
        attachments = {}
        if '_attachments' in self.data.keys():
            if self.data['_attachments'] is not None:
                for attachment in self.data['_attachments']:
                    filename = attachment['filename'].split('/')[-1]
                    downloadurl = attachment['download_url']
                    mimetype = attachment['mimetype']
                    attachments[filename] = {'url': downloadurl, 'mimetype': mimetype}
        self.attachments = attachments
    
    def update_kobo_data_bulk(self, submission_ids, field_, value_):
        """update field of kobo submissions"""
        payload = {
            "submission_ids": [str(id_) for id_ in submission_ids],
            "data": {field_: value_}
        }
        update_result = requests.patch(
            f'https://kobo.ifrc.org/api/v2/assets/{self.asset}/data/bulk/',
            data={'payload': json.dumps(payload)},
            headers={'Authorization': f'Token {self.token}'},
            params={'format': 'json'}
        ).json()
        if 'successes' not in update_result.keys():
            raise RuntimeError("Kobo update failed.")
        elif update_result['successes'] != len(submission_ids):
            raise RuntimeError(f"Kobo update failed for {update_result['failures']} submissions.")
    
    def get_kobo_data_bulk(self):
        kobo_headers = {'Authorization': f"Token {self.token}", "Content-Type": "application/json"}
        res = requests.get(f"https://kobo.ifrc.org/api/v2/assets/{self.asset}/data.json",
                           headers=kobo_headers)
        return res.json()['results']
