export interface ISocialMedia {
    displayText: string;
    imageSrc: string;
    imageAlt: string;
    href: string;
}

export const SocialMediaData: ISocialMedia[] = [
    {
        displayText: 'GitHub',
        imageSrc: 'ico/github-logo.png',
        imageAlt: 'github',
        href: 'https://github.com'
    }
];
