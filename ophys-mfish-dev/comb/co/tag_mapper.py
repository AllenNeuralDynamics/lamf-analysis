



class TagMapper():
    """Given list of tags parse into key:value pairs"""

    def __init__(self, tags):
        self.tags = tags
        self.tag_dict = self._parse_tags()

    # example tags
#     ['multiplane-ophys',
#    'pipeline-v3.0',
#    'suite2p-segmentation-cellpose',
#    'decrosstalk-roi-images-ica',
#    'suite2p-motion-correction',
#    'oasis-event-detection',
#    'derived',
#    '645814',
#    'fe717f53-c918-48d7-972c-01346431bd87',
#    '1223719514']

    def _parse_tags(self):
        tag_dict = {'unknown': []}
        for tag in self.tags:
            #if tag has 'pipeline'
            if 'pipeline' in tag:
                key = 'pipeline_version'
                value = tag
                tag_dict[key] = value
            elif 'multiplane' in tag:
                key = 'type' # TODO: type correct?
                value = tag
                tag_dict[key] = value
            elif 'segmentation' in tag:
                key = 'segmentation'
                value = tag
                tag_dict[key] = value
            elif 'decrosstalk' in tag:
                key = 'decrosstalk'
                value = tag
                tag_dict[key] = value
            elif 'motion' in tag:
                key = 'motion_correction'
                value = tag
                tag_dict[key] = value
            elif 'event' in tag:
                key = 'event_detection'
                value = tag
                tag_dict[key] = value
            # check for derived or raw in tag
            elif tag == 'derived' or tag == 'raw':
                key = 'data_type'
                value = tag
                tag_dict[key] = value
            elif len(tag) == 6 and tag.isdigit():
                key = 'mouse_id'
                value = tag
                tag_dict[key] = value
            elif len(tag) == 36 and '-' in tag:
                key = 'raw_asset_id'
                value = tag
                tag_dict[key] = value
            else:
                key = 'unknown'
                value = tag
                tag_dict[key].append(value)

            # if len of unkown = len of tags, then no tags
            if len(tag_dict['unknown']) == len(self.tags):
                print("Tag parser found no recognized tags")

        return tag_dict


