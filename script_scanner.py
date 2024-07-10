import cv2
import discord
from discord.ext import commands
from io import BytesIO
import Levenshtein
import numpy
import os
import pytesseract
import requests
import sys

A4_HEIGHT_PIXELS = 2970
A4_WIDTH_PIXELS = 2100

character_mapping = {}

def load_character_mapping():
    """Load the character data from the file on disk"""

    character_tsv_file = 'characters.tsv'
    
    with open( character_tsv_file, 'r' ) as file:
        for line in file:
            columns = line.strip().split( '\t' )
            if len(columns) == 2:
                key = columns[ 0 ]
                value = columns[ 1 ]
                character_mapping[ key ] = value


def remove_color( image, threshold = 30 ):
    """Replace any colored pixels with white"""
    
    # Calculate the absolute difference between the RGB channels
    diff_rg = numpy.abs( image[ :, :, 0 ] - image[ :, :, 1 ] )
    diff_rb = numpy.abs( image[ :, :, 0 ] - image[ :, :, 2 ] )
    diff_gb = numpy.abs( image[ :, :, 1 ] - image[ :, :, 2 ] )
    
    # Create a mask for pixels that are not close to being grey
    non_grey_mask = ( diff_rg > threshold ) | ( diff_rb > threshold ) | ( diff_gb > threshold )
    
    # Conert the image to grayscale, setting colored pixels to white using the mask
    gray_image = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY )
    gray_image[ non_grey_mask ] = 255
    return gray_image

def normalize_height( image, height ):
    """Scale an image to a particular height, maintaining the original aspect ratio"""
    ( h, w ) = image.shape[ :2 ]
    aspect_ratio = w / h
    ( h_norm, w_norm ) = ( height, int( height * aspect_ratio ) )
    return cv2.resize( image, ( w_norm, h_norm ) )

def crop_to_character_names( script_image ):
    """Given an image of a full script, this will crop it to just the list of character names"""
    
    width = numpy.size( script_image, 1 )

    if width < 1500:
        raise RuntimeError( f"Image is narrower than expected. width = {width}." )

    # Assuming an A4 sized image, this is where the character names should be
    ( x_min, x_max ) = ( 180, 414 )
    ( y_min, y_max ) = ( 140, 2925 )

    # Account for images that are not A4 sized by adjusting the x-coordinates
    left_margin_size = int( ( width - A4_WIDTH_PIXELS ) * 0.5 )
    x_min += left_margin_size
    x_max += left_margin_size

    return script_image[ y_min:y_max, x_min:x_max ]

def map_scanned_character_names_to_json_equivalent( names ):
    """Convert from printed character names to json names (e.g. "Scarlet Woman" -> "scarlet_woman")"""

    json_characters = []
    for name in names:
        json_character = character_mapping.get( name )

        # If we found an exact match, we are done
        if json_character != None:
            json_characters.append( json_character )
            continue

        # Otherwise, compare the scanned text with all possible character names to see if there is one that is similar
        dist = lambda x: Levenshtein.distance( name, x )
        ( closest_match, distance ) = min( ( ( key, dist( key )) for key in character_mapping.keys()), key = lambda x: x[1] )
        if distance <= 3:
            json_characters.append( character_mapping.get( closest_match ) )
            continue

        # If not, we have failed
        raise RuntimeError( f"Failed to find a match for character name: {name}" )

    return json_characters

def extract_character_names( script_image ):
    """Given a processed script image, return a list of characters"""

    character_names_image = crop_to_character_names( script_image )

    # We remove any colored parts to erase Jinx symbols as these can be detected as text.
    character_names_image = remove_color( character_names_image )

    # By default OpenCV stores images in BGR format and since pytesseract assumes RGB format,
    # we need to convert from BGR to RGB format/mode:
    character_names_image = cv2.cvtColor( character_names_image, cv2.COLOR_BGR2RGB )

    character_names_raw = ""

    try:
        character_names_raw = pytesseract.image_to_string(
            character_names_image,
            config = "--psm 4",
            timeout = 3,
        )
    except RuntimeError as timeout_error:
        raise RuntimeError( "Image processing took too long when scanning character names" ) from timeout_error

    character_names = character_names_raw.strip().split( "\n" )

    # Remove any empty strings from the list
    character_names = [ line for line in character_names if line.strip() != '' ]

    return map_scanned_character_names_to_json_equivalent( character_names )

def extract_script_meta_data( script_image ):
    """Given an image of a script, extract the name of the script and the author name"""
    script_name = ""
    author = ""

    scaled_image_width = numpy.size( script_image, 1 )
    title_image = script_image[ 50:100, 0:scaled_image_width ]
    title_image = cv2.cvtColor( title_image, cv2.COLOR_BGR2RGB )

    try:
        scanned_title_raw = pytesseract.image_to_string(
            title_image,
            #config = "--psm 7",
            timeout = 3,
        )
    except RuntimeError as timeout_error:
        raise RuntimeError( "Image processing took too long when scanning script metadata" ) from timeout_error

    title_split = scanned_title_raw.strip().split( "by", 1 )
    if len( title_split ) == 2:
        script_name = title_split[ 0 ].strip()
        author = title_split[ 1 ].strip()
    else:
        script_name = scanned_title_raw

    return ( script_name, author )

def combine_to_json_string( characters, script_name, author ):
    """Given all the script data, convert it into the standard JSON format"""
    
    json = f'[{{"id":"_meta","author":"{author}","name":"{script_name}"}}'
    for name in characters:
        json += f',"{name}"'
    json +=']'

    # Remove and line breaks or carriage returns
    json = json.replace( "\n", "" ).replace( "\r", "" )

    return json

def script_image_to_json( input_image ):
    """Given an image of a script, convert it into the standard JSON format"""
    processed_script_image = normalize_height( input_image, A4_HEIGHT_PIXELS )
    character_names = extract_character_names( processed_script_image )
    ( script_name, author ) = extract_script_meta_data( processed_script_image )
    json = combine_to_json_string( character_names, script_name, author )
    return ( script_name, author, json )

def bytesio_to_cv2_image( bytesio ):
    image_bytes = bytesio.read()
    np_array = numpy.frombuffer( image_bytes, numpy.uint8 )
    return cv2.imdecode( np_array, cv2.IMREAD_COLOR )

def get_referenced_image( message ):
    """Given a Discord message context, return the first image attached to it."""

    if message.attachments:
        attachment = message.attachments[ 0 ]
        if attachment.content_type.startswith( "image/" ):
            return attachment

async def get_referenced_image_or_parent( ctx ):
    """
    Given a Discord message context, return the image attachment of either that image or, if it
    does not have one, the image attached to the message it is replying to.
    """

    # Return the image attached to this message
    if ( image := get_referenced_image ( ctx.message ) ) is not None:
        return image

    # Or return the image attached to the parent message
    if ctx.message.reference:
        parent_message = await ctx.channel.fetch_message( ctx.message.reference.message_id )
        if ( image := get_referenced_image ( parent_message ) ) is not None:
            return image

async def process_json_request( ctx ):
    attached_image = await get_referenced_image_or_parent( ctx )

    if attached_image is None:
        await ctx.reply( "Please attach a script image." )
        return

    try:
        response = requests.get( attached_image.url )
        image = bytesio_to_cv2_image( BytesIO( response.content ) )
    except Exception:
        await ctx.reply( "Something went wrong." )
        return

    try:
        ( script_name, author, json ) = script_image_to_json( image )
        reply_body = ""
        if len( script_name ) > 0:
            if len( author ) > 0:
                reply_body = f"{script_name} by {author}\n"
            else:
                reply_body = f"{script_name}\n"
        reply_body += f"```json\n{ json }\n```"
        await ctx.reply( reply_body )
    except Exception:
        await ctx.reply( "Something went wrong." )
        return


if __name__ == "__main__":
    try:
        load_character_mapping()
    except Exception as e:
        print( "Error loading character data" )
        sys.exit(1)

    intents = discord.Intents.default()
    intents.messages = True
    intents.message_content = True
    bot = commands.Bot( command_prefix='!', intents = intents )

    @bot.command()
    async def json( ctx ):
        await process_json_request( ctx )

    bot.run( os.environ[ 'JSON_BOT_TOKEN' ] )
