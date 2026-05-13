def add_domain_metadata_to_img_info(domain: str, img_meta: dict, img_info: dict):
    """Adds domain-specific metadata to the image information dictionary."""
    if domain == "Chemistry":
        if 'domain_metadata.molecules' in img_meta:
            img_info['Molecules'] = img_meta['domain_metadata.molecules']

        if 'domain_metadata.reactions' in img_meta:
            img_info['Reactions'] = img_meta['domain_metadata.reactions']
    return img_info


def format_domain_metadata(domain: str, img_info_list: list):
    """Formats domain-specific metadata for text context."""
    domain_metadata = ""
    if domain == "Chemistry":
        for img_info in img_info_list:
            if 'Molecules' in img_info:
                domain_metadata += f"Molecules found in {img_info['Paper']}:\n{img_info['Molecules']}\n\n"
            if 'Reactions' in img_info:
                domain_metadata += f"Reactions found in {img_info['Paper']}:\n{img_info['Reactions']}\n\n"
    return domain_metadata