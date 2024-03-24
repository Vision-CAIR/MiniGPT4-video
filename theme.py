import gradio as gr
# https://www.gradio.app/docs/themes

minigptlv_style = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c50="#ff339c",
        c100="#791aff", 
        c200="#ff339c",
        c300="#ff339c", 
        c400="#ff339c", 
        c500="3384FF", 
        c600="#ff339c", 
        c700="#ff339c", 
        c800="#ff339c",
        c900="#ff339c", 
        c950="#ff339c", 
        name="lighter_blue",
    ),
    secondary_hue=gr.themes.Color(
        c50="#ff339c",
        c100="#ff339c", 
        c200="#ff339c",
        c300="#ff339c",
        c400="#ff339c", 
        c500="#ff339c", 
        c600="#ff339c", 
        c700="#ff339c", 
        c800="#ff339c", 
        c900="#ff339c",
        c950="#ff339c", 
    ),
    neutral_hue=gr.themes.Color(
        c50="#ff339c", 
        c100="#FFFFFF", 
        c200="#3384FF", 
        c300="#ff339c",
        c400="#FFFFFF", 
        c500="#FFFFFF", 
        c600="#ff339c", 
        c700="#192423",
        c800="#cccdde", 
        c900="#ff339c", 
        c950="#ff339c",
        name="dark_scale",
    ),
    radius_size=gr.themes.sizes.radius_sm,
).set(
    button_primary_text_color="#ff339c",
    button_primary_background_fill="#ff339c",
    button_primary_background_fill_dark="#FFFFFF",
    button_primary_border_color_dark="#FFFFFF", 
    button_primary_text_color_dark="#000000", 
    button_secondary_background_fill="#ff339c",
    button_secondary_background_fill_hover="#40c928",
    button_secondary_background_fill_dark="#ff339c",  
     button_secondary_background_fill_hover_dark="#40c928",
    button_secondary_text_color="white",
    button_secondary_text_color_dark="#white", 
    block_title_background_fill_dark="#1a94ff",
    block_label_background_fill_dark="#1a94ff", 
    input_background_fill="#999999", 
    background_fill_primary="#1e1d1f", 
    background_fill_primary_dark="#1e1d1f", 
)

# Define custom CSS
custom_css = """
    /* Custom CSS for Gradio interface */
    .input-box {
        font-family: Arial, sans-serif;
        background-color: #F0F0F0;
        border: 1px solid #CCCCCC;
    }

    .output-box {
        font-family: Arial, sans-serif;
        background-color: #FFFFFF;
        border: 1px solid #CCCCCC;
    }

    .checkbox {
        color: #464646;
    }

    .textbox {
        width: 100%;
    }

    .output-image {
        border: 1px solid #CCCCCC;
    }
    """
    
text_css = """
h1 {
    text-align: center;
    display:block;
    font-size: 45px;
}
h5 {
    text-align: center;
    display:block;
}
"""